import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from collections import namedtuple
from datetime import datetime
import json
import os
import sys
import pickle

from pathlib import Path

from dm_env import specs
import matplotlib.pyplot as plt
import numpy as np
from tap import Tap
import torch
from tqdm import tqdm
import wandb

import drqv2.dmc as dmc
from drqv2.drqv2 import DrQV2Agent
import drqv2.utils as drqv2_utils
from drqv2.replay_buffer import ReplayBuffer, ReplayBufferStorage, make_replay_loader
from drqv2.video import VideoRecorder

from policy_learning.envs import (
    ImageMetaworldEnv,
    ImageOnlineCustomRewardMetaworldEnv,
    LowDimMetaworldEnv,
    LowDimOnlineCustomRewardMetaworldEnv,
    LowDimStateCustomRewardFromImagesMetaworldEnv,
    TcnImageMetaworldEnv
)
from policy_learning.soil_utils import SOIL
from reward_extraction.reward_functions import LearnedImageRewardFunction, LearnedRewardFunction, check_and_generate_expert_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MetaworldWorkspaceV2:
    REL_REPO_ROOT_DIR = "policy_learning"
    PROPRIOCEPTION_DIM = 4

    def __init__(
        self,
        env_str: str,
        exp_str: str,
        seed: int = 0,
        discount_rate: float = 0.99,
        num_seed_frames: int = 5000,
        num_train_frames: int = 1500000,
        save_eval_video: bool = True,
        save_train_video: bool = True,
        with_online_learned_reward_fn: bool = False,
        drqv2_feature_dim: int = 50,
        train_classifier_with_mixup: bool = True,
        do_film_layer: bool = True,
        camera_name: str = None,
        with_high_res_img: bool = False,
        with_ppc: bool = False,
        rl_on_state: bool = True,
        lrf_on_state: bool = True,
        repo_root: str = None,
        experiment_folder: str = "experiments",
        train_gail: bool = False,
        train_airl: bool = False,
        train_vice: bool = False,
        train_soil: bool = False,
        take_log_reward: bool = False,
        take_d_ratio: bool = False,
        lgn_multiplier: float = 1.0,
        refresh_reward: bool = False,
        disable_classifier: bool = False,
        train_tcn: bool = False,
    ):
        curr_work_directory = str(Path.cwd())
        if 'gridsan' in curr_work_directory:
            self.wandb_mode = "offline"
        else:
            # self.wandb_mode = "offline" # let's just always have it offline and manually sync
            self.wandb_mode = "online"

        if self.REL_REPO_ROOT_DIR == "policy_learning":
            if train_gail:
                project_name = "rewardlearningvid-metaworld-gail"
            elif train_airl:
                project_name = "rewardlearningvid-metaworld-airl"
            elif train_vice:
                project_name = "rewardlearningvid-metaworld-vice"
            elif train_soil:
                project_name = "rewardlearningvid-metaworld-soil"
            elif train_tcn:
                project_name = "rewardlearningvid-metaworld-tcn"
            elif not with_online_learned_reward_fn:
                project_name = "rewardlearningvid-metaworld-vanilla"
            elif disable_classifier:
                project_name = "rewardlearningvid-metaworld-justrank"
            else:
                project_name = "rewardlearningvid-metaworld"

        ##############################################
        '''
        here is the default training config. migrate options
        into the arguments for this object as the need arises
        '''
        config = namedtuple("Config",
            ["env_str", "seed", "discount_rate", "num_seed_frames", "num_train_frames", "save_eval_video",
            "save_train_video", "eval_every_frames", "num_eval_episodes", "save_snapshot"]
        )
        config.env_str = env_str
        config.discount_rate = discount_rate
        config.num_seed_frames = num_seed_frames
        config.num_train_frames = num_train_frames
        config.save_eval_video = save_eval_video
        config.save_train_video = save_train_video
        config.eval_every_frames = 20000
        config.num_eval_episodes = 10
        config.save_snapshot = True
        self.cfg = config

        # replay buffer
        rb_config = namedtuple("RBConfig",
            ["rb_size", "num_workers", "nstep", "batch_size"]
        )
        rb_config.rb_size = 1000000
        rb_config.num_workers = 4 # cranked up the RAM and CPU cores on .jaynes.yml so don't think this is an issue anymore
        rb_config.nstep = 1
        rb_config.batch_size = 256
        self.rb_config = rb_config

        # agent
        agent_config = namedtuple("AgentConfig",
            ["lr", "critic_target_tau", "update_every_steps", "use_tb", "num_expl_steps", "hidden_dim",
            "feature_dim", "stddev_schedule", "stddev_clip"]
        )
        agent_config.lr = 1e-4
        agent_config.critic_target_tau = 0.01
        agent_config.update_every_steps = 2
        agent_config.use_tb = True # ensure that metrics are returned
        agent_config.num_expl_steps = 2000
        agent_config.hidden_dim = 1024
        agent_config.feature_dim = drqv2_feature_dim
        agent_config.stddev_schedule = 'linear(1.0,0.1,500000)'
        agent_config.stddev_clip = 0.3
        self.agent_config = agent_config
        ##############################################

        self.env_str = env_str
        self.with_online_learned_reward_fn = with_online_learned_reward_fn
        self.train_classifier_with_mixup = train_classifier_with_mixup
        self.do_film_layer = do_film_layer
        self.rl_on_state = rl_on_state
        self.lrf_on_state = lrf_on_state
        self.with_high_res_img = with_high_res_img
        self.camera_name = camera_name
        self.with_ppc = with_ppc

        self.take_log_reward = take_log_reward
        self.take_d_ratio = take_d_ratio
        self.refresh_reward = refresh_reward
        self.lgn_multiplier = lgn_multiplier

        self.disable_classifier = disable_classifier

        # modifications to run baselines
        self.disable_ranking = False
        self.train_gail = train_gail
        self.train_airl = train_airl
        self.train_vice = train_vice
        self.train_soil = train_soil
        self.train_tcn = train_tcn
        assert (int(train_gail) + int(train_airl) + int(train_vice) + int(train_soil) + int(train_tcn)) <= 1
        if self.train_gail or self.train_airl or self.train_vice:
            self.disable_ranking = True
        if self.train_soil or self.train_tcn:
            assert not self.with_online_learned_reward_fn
            self.with_ppc = False
        if self.train_gail or self.train_airl or self.train_vice or self.train_soil or self.train_tcn:
            # these should only be used with our method
            self.take_log_reward = False
            self.take_d_ratio= False

        if disable_classifier:
            assert (int(train_gail) + int(train_airl) + int(train_vice) + int(train_soil) + int(train_tcn)) == 0
            assert not self.disable_ranking


        if repo_root is None:
            self.repo_root = Path.cwd()
        else:
            self.repo_root = Path(repo_root)

        self.work_dir = self.repo_root / self.REL_REPO_ROOT_DIR / experiment_folder / exp_str
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir, exist_ok=True)

        print(f'repo_root: {self.repo_root}')
        print(f'workspace: {self.work_dir}')

        wandb.init(project=project_name, name=exp_str, mode=self.wandb_mode, dir=self.work_dir, settings=wandb.Settings(start_method='fork'))
        print(f"wandb is using {wandb.run.name} in {self.wandb_mode} mode")


        drqv2_utils.set_seed_everywhere(seed)
        self.setup()

        self.agent = DrQV2Agent(
            obs_shape=self.train_env.observation_spec().shape,
            action_shape=self.train_env.action_spec().shape,
            device=device,
            lr=self.agent_config.lr,
            feature_dim=self.agent_config.feature_dim,
            hidden_dim=self.agent_config.hidden_dim,
            critic_target_tau=self.agent_config.critic_target_tau,
            num_expl_steps=self.agent_config.num_expl_steps,
            update_every_steps=self.agent_config.update_every_steps,
            stddev_schedule=self.agent_config.stddev_schedule,
            stddev_clip=self.agent_config.stddev_clip,
            use_tb=self.agent_config.use_tb,
            with_ppc=self.with_ppc,
            proprioception_dim=self.PROPRIOCEPTION_DIM
        )

        if self.train_soil:
            expert_data_path = f"{self.work_dir}/expert_data.hdf"
            self.soil = SOIL(
                state_shape=self.train_env.observation_spec().shape,
                action_shape=self.train_env.action_spec().shape,
                rb_loader=self.replay_loader,
                state_dset_path=expert_data_path
            )

        self.timer = drqv2_utils.Timer()
        self._global_step = 0
        self._global_episode = 0


    def setup(self):
        assert self.rl_on_state == self.lrf_on_state

        # create envs
        if self.rl_on_state:
            rl_dummy_env = LowDimMetaworldEnv(self.env_str)
        else:
            rl_dummy_env = ImageMetaworldEnv(self.env_str, camera_name=self.camera_name, high_res_env=self.with_high_res_img)

        if self.lrf_on_state:
            lrf_dummy_env = LowDimMetaworldEnv(self.env_str)
        else:
            lrf_dummy_env = ImageMetaworldEnv(self.env_str, camera_name=self.camera_name, high_res_env=self.with_high_res_img)

        # create replay buffer
        data_specs = (rl_dummy_env.observation_spec(),
                      rl_dummy_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'og_reward'),
                      specs.Array((1,), np.float32, 'discount'),
                      specs.BoundedArray(shape=(39,),
                                        dtype=np.float32,
                                        minimum=-np.inf,
                                        maximum=np.inf,
                                        name='metaworld_state_obs'),
                      )

        self.lrf_obs_key = "observation"
        if self.with_online_learned_reward_fn:
            if self.rl_on_state != self.lrf_on_state:
                spec = lrf_dummy_env.observation_spec()
                spec._name = "lowdimstate" if self.lrf_on_state else "image"
                self.lrf_obs_key = spec.name
                data_specs = data_specs + (spec,)

        replay_storage_path = self.work_dir / 'buffer'
        self.replay_storage = ReplayBufferStorage(data_specs, replay_storage_path)

        self.replay_loader = make_replay_loader(
            replay_storage_path, self.rb_config.rb_size,
            self.rb_config.batch_size, self.rb_config.num_workers,
            self.cfg.save_snapshot, self.rb_config.nstep, self.cfg.discount_rate)
        self._replay_iter = None

        # we use the expert data for BC initialization both with and without the LRF
        check_and_generate_expert_data(str(self.work_dir), self.env_str, lrf_dummy_env, 100, False)

        # for the reward function training. affects which environment we instantiate
        if self.with_online_learned_reward_fn:
            self.train_lrf_for_steps = 20
            self.train_lrf_frequency_steps = 2000

            # anything with a lrf needs this
            self.rb_for_reward_fn = ReplayBuffer(
                replay_storage_path,
                self.rb_config.rb_size,
                1,
                self.rb_config.nstep,   # doesn't really matter
                self.cfg.discount_rate, # doesn't really matter
                fetch_every=1,          # refresh the replay buffer everytime we call _sample()
                save_snapshot=True      # if this isn't true, replay buffer starts getting deleted which wouldn't be super
            )

            # figure out which environment we are instantiating
            if self.lrf_on_state:
                assert self.lrf_on_state == self.rl_on_state
                self.learned_reward_function = LearnedRewardFunction(
                    obs_size=lrf_dummy_env.observation_spec().shape[0],
                    exp_dir=self.work_dir,
                    replay_buffer=self.rb_for_reward_fn,
                    train_classify_with_mixup=self.train_classifier_with_mixup,
                    add_state_noise=True,
                    rb_buffer_obs_key=self.lrf_obs_key,
                    disable_ranking=self.disable_ranking,
                    disable_classifier=self.disable_classifier,
                    train_classifier_with_goal_state_only=self.train_vice,
                )
                self.train_env = LowDimOnlineCustomRewardMetaworldEnv(self.env_str, lrf=self.learned_reward_function, airl_style_reward=self.train_airl, take_log_reward=self.take_log_reward, take_d_ratio=self.take_d_ratio, lgn_multiplier=self.lgn_multiplier)
                self.eval_env = LowDimOnlineCustomRewardMetaworldEnv(self.env_str, lrf=self.learned_reward_function, airl_style_reward=self.train_airl, take_log_reward=self.take_log_reward, take_d_ratio=self.take_d_ratio, lgn_multiplier=self.lgn_multiplier)
            else:
                self.learned_reward_function = LearnedImageRewardFunction(
                    obs_size=lrf_dummy_env.observation_spec().shape,
                    exp_dir=self.work_dir,
                    replay_buffer=self.rb_for_reward_fn,
                    train_classify_with_mixup=self.train_classifier_with_mixup,
                    add_state_noise=True,
                    rb_buffer_obs_key=self.lrf_obs_key,
                    disable_ranking=self.disable_ranking,
                    disable_classifier=self.disable_classifier,
                    train_classifier_with_goal_state_only=self.train_vice,
                    do_film_layer=self.do_film_layer,
                )

                if self.rl_on_state:
                    self.train_env = LowDimStateCustomRewardFromImagesMetaworldEnv(self.env_str, camera_name=self.camera_name, high_res_env=self.with_high_res_img, lrf=self.learned_reward_function, take_log_reward=self.take_log_reward, take_d_ratio=self.take_d_ratio, lgn_multiplier=self.lgn_multiplier)
                    self.eval_env = LowDimStateCustomRewardFromImagesMetaworldEnv(self.env_str, camera_name=self.camera_name, high_res_env=self.with_high_res_img, lrf=self.learned_reward_function, take_log_reward=self.take_log_reward, take_d_ratio=self.take_d_ratio, lgn_multiplier=self.lgn_multiplier)
                else:
                    self.train_env = ImageOnlineCustomRewardMetaworldEnv(self.env_str, camera_name=self.camera_name, high_res_env=self.with_high_res_img, lrf=self.learned_reward_function, airl_style_reward=self.train_airl, take_log_reward=self.take_log_reward, take_d_ratio=self.take_d_ratio, lgn_multiplier=self.lgn_multiplier)
                    self.eval_env = ImageOnlineCustomRewardMetaworldEnv(self.env_str, camera_name=self.camera_name, high_res_env=self.with_high_res_img, lrf=self.learned_reward_function, airl_style_reward=self.train_airl, take_log_reward=self.take_log_reward, take_d_ratio=self.take_d_ratio, lgn_multiplier=self.lgn_multiplier)
        elif self.train_tcn:
            print(f"using vanilla environments with TCN reward override")
            expert_data_path = f"{self.work_dir}/expert_data.hdf"
            tcn_model_path = f"{self.repo_root}/tcn/models/{self.env_str}/tcn_embedder.pt"

            if self.rl_on_state:
                assert False # never implemented
            else:
                self.train_env = TcnImageMetaworldEnv(
                    self.env_str,
                    camera_name=self.camera_name,
                    high_res_env=self.with_high_res_img,
                    expert_data_path=expert_data_path,
                    tcn_model_path=tcn_model_path
                )
                self.eval_env = TcnImageMetaworldEnv(
                    self.env_str,
                    camera_name=self.camera_name,
                    high_res_env=self.with_high_res_img,
                    expert_data_path=expert_data_path,
                    tcn_model_path=tcn_model_path
                )
        else:
            print(f"using vanilla environments!")
            if self.rl_on_state:
                self.train_env = LowDimMetaworldEnv(self.env_str)
                self.eval_env = LowDimMetaworldEnv(self.env_str)
            else:
                self.train_env = ImageMetaworldEnv(self.env_str, camera_name=self.camera_name, high_res_env=self.with_high_res_img)
                self.eval_env = ImageMetaworldEnv(self.env_str, camera_name=self.camera_name, high_res_env=self.with_high_res_img)

        # setup video recorders
        if  self.with_high_res_img:
            render_size = 224
        else:
            render_size = 96
        self.video_recorder = VideoRecorder(self.work_dir if self.cfg.save_eval_video else None, folder_name="eval_video", render_size=render_size, camera_name=self.camera_name)
        self.train_video_recorder = VideoRecorder(self.work_dir if self.cfg.save_train_video else None, folder_name="train_video", render_size=render_size, camera_name=self.camera_name)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self, save_str: str = None, num_eval_episodes: int = None):
        step, episode = 0, 0

        if num_eval_episodes is None:
            eval_until_episode = drqv2_utils.Until(self.cfg.num_eval_episodes)
        else:
            eval_until_episode = drqv2_utils.Until(num_eval_episodes)

        og_total_rewards = []
        total_rewards = []
        successes = []
        succeededs = []

        self.video_recorder.init(self.eval_env, enabled=True)
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            per_episode_reward = 0
            per_episode_og_reward = 0
            succeeded = False
            while not time_step.last():
                with torch.no_grad(), drqv2_utils.eval_mode(self.agent):
                    proprioception_dim = 4 # metaworld hand xyz gripper (not used unless image observations)
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True,
                                            proprioception=time_step.metaworld_state_obs[:proprioception_dim])
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)

                per_episode_reward += time_step.reward
                per_episode_og_reward += self.eval_env.get_last_received_reward()

                step += 1
                succeeded |= int(time_step.success)

            total_rewards.append(per_episode_reward)
            og_total_rewards.append(per_episode_og_reward)

            successes.append(time_step.success)
            succeededs.append(succeeded)
            episode += 1

        if save_str is None:
            self.video_recorder.save(f'{self.global_frame}.mp4')
        else:
            self.video_recorder.save(f'{save_str}.mp4')

        rewards_dict = {
            "rewards": total_rewards,
            "og_rewards": og_total_rewards,
            "success": successes,
            "succeeded": succeededs,
        }

        if self.video_recorder.save_dir is not None:
            rewards_json = json.dumps(rewards_dict, indent=4)
            if save_str is None:
                json_path = self.video_recorder.save_dir / f'{self.global_frame}.json'
            else:
                json_path = self.video_recorder.save_dir / f'{save_str}.json'
            with open(json_path, "w") as outfile:
                outfile.write(rewards_json)

        total_time = self.timer.total_time()
        metrics={
            'eval/episode_reward': np.mean(total_rewards),
            'eval/og_episode_reward': np.mean(og_total_rewards),
            'eval/episode_length': step / episode,
            'eval/episode': self.global_episode,
            'eval/success_rate': np.mean(successes),
            'eval/succeeded_rate': np.mean(succeededs),
            'eval/step': self.global_step,
            'eval_total_time': total_time,
        }
        if save_str is None:
            wandb.log(metrics, step=self.global_frame)


    def train(self):
        # predicates
        train_until_step = drqv2_utils.Until(self.cfg.num_train_frames)
        seed_until_step = drqv2_utils.Until(self.cfg.num_seed_frames)
        eval_every_step = drqv2_utils.Every(self.cfg.eval_every_frames)

        episode_step, episode_reward = 0, 0
        og_episode_reward = 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)

        metrics = None
        is_train_recording = False
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1

                # some bookkeeping around recording train videos
                if is_train_recording:
                    self.train_video_recorder.save(f'{self.global_frame}.mp4')
                    is_train_recording = False

                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step
                    metrics={
                        'train/fps': episode_frame / elapsed_time,
                        'train/total_time': total_time,
                        'train/episode_reward': episode_reward,
                        'train/og_episode_reward': og_episode_reward,
                        'train/episode_length': episode_frame,
                        'train/episode': self.global_episode,
                        'train/buffer_size': len(self.replay_storage),
                        'train/step': self.global_step,
                    }
                    if self._global_episode % 10 == 0:
                        wandb.log(metrics, step=self.global_frame)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)

                if self._global_episode % 100 == 0:
                    is_train_recording = True
                    self.train_video_recorder.init(self.train_env, enabled=True)

                # try to save snapshot
                if self._global_episode % 100 == 0 and self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0
                og_episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.eval()

            # sample action
            with torch.no_grad(), drqv2_utils.eval_mode(self.agent):
                proprioception_dim = 4 # hand xyz gripper (not used unless images)
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False,
                                        proprioception=time_step.metaworld_state_obs[:proprioception_dim])
            # try to update the agent
            if not seed_until_step(self.global_step):
                # do SOIL things
                if self.train_soil:
                    self.soil.update_inverse_model()
                    imitation_loss, _ = self.soil.calculate_policy_loss(self.agent, self.global_step - self.cfg.num_seed_frames)
                    metrics = self.agent.update_actor_with_other_loss(imitation_loss)
                else:
                    if self.with_online_learned_reward_fn:
                        # note: we don't have goal images for metaworld, just goal states
                        metrics = self.agent.update(
                            self.replay_iter, self.global_step,
                            lrf=self.learned_reward_function, goal_image=None,
                            airl_style_reward=self.train_airl, take_log_reward=self.take_log_reward, take_d_ratio=self.take_d_ratio, lgn_multiplier=self.lgn_multiplier,
                            refresh_reward=self.refresh_reward,
                        )
                    else:
                        metrics = self.agent.update(self.replay_iter, self.global_step)

                if self.global_step % 1000 == 0:
                    wandb.log(metrics, step=self.global_frame)

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            og_episode_reward += self.train_env.get_last_received_reward()
            self.replay_storage.add(time_step)

            if is_train_recording:
                self.train_video_recorder.record(self.train_env)

            # debug heartbeat
            if self._global_step % 5000 == 0:
                print(f"-----[{self._global_step}]---------------")
                print(f"curent job is using this directory: {self.work_dir}")
                print(f"with this wandb run name: {wandb.run.name} in {self.wandb_mode} mode now")

            # we don't have to train the lrf if we aren't using the classifier
            if not self.disable_classifier:
                # periodically retrain the learned reward function, don't start training till agent is seeded
                if self.with_online_learned_reward_fn:
                    if not seed_until_step(self.global_step):
                        # ping the replay buffer dataset so it gets periodically refreshed as well
                        _ = self.rb_for_reward_fn._sample()

                        if self._global_step % self.train_lrf_frequency_steps == 0 and self._global_step > 0:
                            self.learned_reward_function.train(self.train_lrf_for_steps)

                # periodically evaluate the learned reward function
                if self._global_step % 200000 == 0 and self._global_step > 0:
                    if self.with_online_learned_reward_fn:
                        self.eval_lrf(iter_num=str(self._global_step))

            episode_step += 1
            self._global_step += 1

        print(f"training done!")
        self.save_snapshot()


    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
        print(f"successfully loaded workspace snapshot at step {self.global_step} from {snapshot}")

    def eval_lrf(self, iter_num: str = "end"):
        assert self.with_online_learned_reward_fn

        from reward_extraction.reward_function_plots import (
            generate_all_traj_rankings,
            generate_counterfactual_plots,
            generate_counterfactual_reward_at_diff_states_plots,
            generate_rb_plots
        )

        self.learned_reward_function.eval_mode()

        analysis_dir = self.work_dir / "analysis" / f"{iter_num}"
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)

        is_lrf_images = not self.lrf_on_state

        print(f"eval through expert trajs as factuals")
        generate_all_traj_rankings(self.learned_reward_function, analysis_dir=analysis_dir, is_input_images=is_lrf_images, on_train_data=True, generate_videos=False)
        generate_all_traj_rankings(self.learned_reward_function, analysis_dir=analysis_dir, is_input_images=is_lrf_images, on_train_data=False, generate_videos=False)

        print(f"eval through expert trajs as counterfactuals")
        generate_counterfactual_plots(self.learned_reward_function, analysis_dir=analysis_dir, is_input_images=is_lrf_images, on_train_data=True)
        generate_counterfactual_plots(self.learned_reward_function, analysis_dir=analysis_dir, is_input_images=is_lrf_images, on_train_data=False)

        print(f"eval with xy plots of hand gripper position")
        generate_counterfactual_reward_at_diff_states_plots(self.env_str, self.learned_reward_function, analysis_dir=analysis_dir, is_input_images=is_lrf_images, on_train_data=True)
        generate_counterfactual_reward_at_diff_states_plots(self.env_str, self.learned_reward_function, analysis_dir=analysis_dir, is_input_images=is_lrf_images, on_train_data=False)

        print(f"eval on replay buffer data")
        generate_rb_plots(self.learned_reward_function, analysis_dir=analysis_dir, rb_obs_key=self.lrf_obs_key, is_input_images=is_lrf_images)

        self.learned_reward_function.save_models(save_dir=analysis_dir)

        self.learned_reward_function.train_mode()


class MetaworldArgumentParser(Tap):
    env_str: str
    seed: int = 0
    use_online_lrf: bool = False

    rl_on_state: bool = False
    lrf_on_state: bool = False

    num_train_frames: int = 1_500_000

    train_gail: bool = False
    train_airl: bool = False
    train_vice: bool = False
    train_soil: bool = False

    train_tcn: bool = False

    discount_rate: float = 0.99

    disable_classifier: bool = False
    turn_off_mixup: bool = False
    turn_off_film: bool = False

    take_log_reward: bool = True
    take_d_ratio: bool = True
    refresh_reward: bool = False
    lgn_multiplier: float = 1.0

def run_train():
    args = MetaworldArgumentParser().parse_args()

    assert args.env_str in [
        "assembly", "drawer-open", "hammer", "door-close", "push",
        "reach", "bin-picking", "button-press-topdown", "door-open"
    ]

    if args.train_gail:
        exp_style = "gail"
    elif args.train_airl:
        exp_style = "airl"
    elif args.train_vice:
        exp_style = "vice"
    elif args.disable_classifier:
        exp_style = "justrank"
    elif args.train_soil:
        exp_style = "soil"
    elif args.use_online_lrf:
        exp_style = "lrf"
    elif args.train_tcn:
        exp_style = "tcn"
    else:
        exp_style = "vanilla"

    if exp_style in ["soil", "vanilla", "tcn"]:
        assert not args.use_online_lrf
    else:
        assert args.use_online_lrf

    if args.rl_on_state:
        exp_substr = exp_style + "LowDim"
        folder_substr = exp_style + "_lowdim"
    else:
        exp_substr = exp_style + "Image"
        folder_substr = exp_style + "_image"

    date_str = datetime.today().strftime('%Y-%m-%d')

    exp_str = f"{date_str}/{folder_substr}/{args.env_str}-{exp_substr}-seed-{args.seed}-dr-{args.discount_rate}-refresh-{args.refresh_reward}-logr-{args.take_log_reward}-d-{args.take_d_ratio}-lgn-{args.lgn_multiplier}"
    if args.turn_off_mixup and args.turn_off_film:
        exp_str += "-noregularization"

    print(f"exp_str: {exp_str}")

    train_classifier_with_mixup = not args.turn_off_mixup
    do_film_layer = not args.turn_off_film

    workspace = MetaworldWorkspaceV2(
        env_str=args.env_str,
        exp_str=exp_str,
        seed=args.seed,
        discount_rate=args.discount_rate,
        with_online_learned_reward_fn=args.use_online_lrf,
        num_train_frames=args.num_train_frames,
        drqv2_feature_dim=128,
        train_classifier_with_mixup=train_classifier_with_mixup,
        do_film_layer=do_film_layer,
        camera_name="left_cap2",
        with_ppc=True,
        rl_on_state=args.rl_on_state,
        lrf_on_state=args.lrf_on_state,
        train_soil=args.train_soil,
        train_gail=args.train_gail,
        train_airl=args.train_airl,
        train_vice=args.train_vice,
        take_log_reward=args.take_log_reward,
        take_d_ratio=args.take_d_ratio,
        lgn_multiplier=args.lgn_multiplier,
        refresh_reward=args.refresh_reward,
        disable_classifier=args.disable_classifier,
        train_tcn=args.train_tcn,
    )

    workspace.train()
    if args.use_online_lrf and not args.disable_classifier:
        workspace.eval_lrf()


if __name__ == '__main__':
    run_train()