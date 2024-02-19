import copy
import functools
import os
from pathlib import Path
import pickle
from typing import List
import warnings
warnings.filterwarnings("ignore")

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns
sns.set_style("white")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from tqdm import tqdm

from drqv2.replay_buffer import ReplayBuffer
from drqv2.video import VideoRecorder

from reward_extraction.data import H5PyTrajDset
from reward_extraction.metaworld_experts import get_expert_policy
from reward_extraction.models import Policy, R3MPolicy, R3MImageGoalPolicy
from policy_learning.envs import LowDimMetaworldEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_expert_data(
    env_str: str,
    env: LowDimMetaworldEnv,
    expert_data_path,
    num_trajs,
    render=False, # useful for debugging behavior of the policy
    generate_videos=False,
):
    # expert policy
    expert_policy = get_expert_policy(env_str)

    # expert data
    dset = H5PyTrajDset(expert_data_path, read_only_if_exists=False)

    # video recorder so we can see what the experts look like
    if generate_videos:
        data_dir = Path(os.path.dirname(expert_data_path))
        video_recorder = VideoRecorder(data_dir, folder_name="experts_test" if "test" in expert_data_path else "experts_train", render_size=96, camera_name="left_cap2")

    # collect the data
    success_count = 0
    pbar = tqdm(num_trajs)
    while success_count < num_trajs:
        time_step = env.reset()

        if generate_videos:
            video_recorder.init(env)

        states = []
        actions = []
        rewards = []
        goals = []
        metaworld_states = []

        states.append(time_step.observation)
        metaworld_states.append(time_step.metaworld_state_obs)
        goals.append(time_step.metaworld_state_obs[-3:])
        while not time_step.last():
            if render:
                env.render() # note this will probably screw with rendering images
            action = expert_policy.get_action(time_step.metaworld_state_obs)
            time_step = env.step(action)

            states.append(time_step.observation)
            actions.append(action)
            rewards.append(time_step.reward)
            goals.append(time_step.metaworld_state_obs[-3:])
            metaworld_states.append(time_step.metaworld_state_obs)

            if generate_videos:
                video_recorder.record(env)

        states = np.expand_dims(np.array(states), axis=0)
        actions = np.expand_dims(np.array(actions), axis=0)
        rewards = np.expand_dims(np.array(rewards), axis=0)
        goals = np.expand_dims(np.array(goals), axis=0)
        metaworld_states = np.expand_dims(np.array(metaworld_states), axis=0)

        if time_step.success:
            dset.add_traj(states, actions, rewards, goals, metaworld_states)
            if generate_videos:
                video_recorder.save(f"{str(success_count).zfill(3)}.mp4")
            success_count += 1
            pbar.update()



def check_and_generate_expert_data(exp_dir: str, env_str: str, env: LowDimMetaworldEnv, num_expert_trajs: int = 100, render: bool = False, generate_test_data: bool = True):
    # if there is no expert data, generate some
    expert_data_path = f"{exp_dir}/expert_data.hdf"
    if not os.path.exists(expert_data_path):
        generate_expert_data(env_str, env, expert_data_path, num_expert_trajs, render=False, generate_videos=True)
    if generate_test_data:
        expert_test_data_path = f"{exp_dir}/expert_test_data.hdf"
        if not os.path.exists(expert_test_data_path):
            generate_expert_data(env_str, env, expert_test_data_path, num_expert_trajs, render=False, generate_videos=True)

def mixup_data(x, y, alpha=1.0, goals=None):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    bs = x.size()[0]

    index = torch.randperm(bs).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    if goals is not None:
        mixed_goals = lam * goals + (1 - lam) * goals[index, :]
        return mixed_x, y_a, y_b, lam, mixed_goals
    else:
        return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class LearnedRewardFunction():
    def __init__(self,
        obs_size: int,
        exp_dir: str,
        replay_buffer: ReplayBuffer,
        train_classify_with_mixup: bool = False,
        add_state_noise: bool = False,
        rb_buffer_obs_key: str = "observation",
        disable_classifier: bool = False, # ablation
        disable_ranking: bool = False, # GAIL / AIRL
        goal_is_metaworld_style: bool = True,
        train_classifier_with_goal_state_only: bool = False, # VICE
        for_rlpd: bool = False,
    ):
        '''
        NOTE: assuming that someone else is responsible for making sure the replay buffer
              seeing fresh data on disk. we just like to have a pointer to it here to access
        '''
        self.exp_dir = exp_dir
        self.horizon = 100
        self.train_classify_with_mixup = train_classify_with_mixup
        self.add_state_noise = add_state_noise
        self.disable_ranking = disable_ranking
        self.disable_classifier = disable_classifier
        self.goal_is_metaworld_style = goal_is_metaworld_style
        self.train_classifier_with_goal_state_only = train_classifier_with_goal_state_only
        if self.train_classifier_with_goal_state_only:
            assert self.disable_ranking

        # training parameters
        self.batch_size = 256
        self.lr = 1e-4

        # network definitions
        hidden_depth = 3
        hidden_layer_size = 1000

        if not self.disable_ranking:
            self.ranking_network = Policy(obs_size, 1, hidden_layer_size, hidden_depth)
            self.ranking_network.to(device)
            self.ranking_optimizer = optim.Adam(list(self.ranking_network.parameters()), lr=self.lr)
        self.same_traj_classifier = Policy(obs_size, 1, hidden_layer_size, hidden_depth)
        self.same_traj_classifier.to(device)
        self.same_traj_optimizer = optim.Adam(list(self.same_traj_classifier.parameters()), lr=self.lr)
        self.bce_with_logits_criterion = torch.nn.BCEWithLogitsLoss()

        # if there is no expert data, generate some
        self.expert_data_path = f"{self.exp_dir}/expert_data.hdf"
        self.expert_test_data_path = f"{self.exp_dir}/expert_test_data.hdf"
        assert os.path.exists(self.expert_data_path)
        self.expert_data_ptr = H5PyTrajDset(self.expert_data_path, read_only_if_exists=True)
        self.expert_data = [d for d in self.expert_data_ptr]
        self.expert_test_data_ptr = H5PyTrajDset(self.expert_test_data_path, read_only_if_exists=True)
        self.expert_test_data = [d for d in self.expert_test_data_ptr]
        self.num_expert_trajs = len(self.expert_data)
        assert self.num_expert_trajs == 100

        # replay buffer
        self.replay_buffer = replay_buffer
        self.rb_buffer_obs_key = rb_buffer_obs_key
        self.for_rlpd = for_rlpd

        # to match the image version
        self.goal_is_image = False

        # bookkeeping
        self.train_step = 0
        self.plot_and_save_frequency = 100
        self.train_steps = []
        self.losses_same_traj = []
        self.losses_std_same_traj = []
        self.running_loss_same_traj = []

        self.seen_on_policy_data = False

        # init and bookkeeping
        self.init_ranking_steps = []
        self.init_ranking_losses = []
        self.init_ranking_losses_std = []
        self.init_ranking()

    def _calculate_reward(self, x, goal=None, airl_style_reward: bool = False, take_log_reward: bool = False, take_d_ratio: bool = False, lgn_multiplier=1.0, eps=1e-6):
        # for low dim, goal should be baked into obs. goal arg is here for API consistency
        with torch.no_grad():
            self.eval_mode()

            if self.disable_ranking:
                mask = torch.sigmoid(self.same_traj_classifier(x)).cpu().numpy()
                reward = mask
            elif self.disable_classifier:
                progress = torch.sigmoid(self.ranking_network(x)).cpu().numpy()
                reward = progress
            else:
                mask = torch.sigmoid(self.same_traj_classifier(x)).cpu().numpy()
                progress = torch.sigmoid(self.ranking_network(x)).cpu().numpy()
                reward = progress * mask

            self.train_mode()

        if airl_style_reward:
            reward_clamped = np.clip(reward, self.eps, 1 - self.eps) # numerical stability
            reward = np.log(reward_clamped) - np.log(1 - reward_clamped)
        elif take_log_reward:
            if self.disable_classifier:
                progress_clamped = np.clip(progress, eps, 1 - eps) # numerical stability
                reward = np.log(progress_clamped)
            else:
                mask_clamped = np.clip(mask, eps, 1 - eps) # numerical stability
                progress_clamped = np.clip(progress, eps, 1 - eps) # numerical stability
                if take_d_ratio:
                    reward = np.log(progress_clamped) + lgn_multiplier * (np.log(mask_clamped) - np.log(1 - mask_clamped))
                else:
                    reward = np.log(progress_clamped) + lgn_multiplier * np.log(mask_clamped)

        return reward

    def _train_step(self):
        '''
        this will train the classifier for a full step, using the ranking steps code to sample from the expert data
        '''
        self.seen_on_policy_data = True

        '''
        sample from expert data (factuals)
        '''
        loss_monotonic, expert_states_t, expert_states_t_np = self._train_ranking_step(do_inference=False)

        '''
        sample another set (1/2 batch size) from expert data (counterfactuals)
        '''
        half_batch_size = self.batch_size // 2
        expert_cf_idxs = np.random.randint(self.num_expert_trajs, size=(half_batch_size,))
        if self.train_classifier_with_goal_state_only:
            expert_cf_t_idxs = np.ones(shape=(half_batch_size,), dtype=np.int64) * (self.horizon - 1)
        else:
            expert_cf_t_idxs = np.random.randint(self.horizon, size=(half_batch_size,))
        expert_cf_states = np.concatenate([self.expert_data[traj_idx][0][t_idx][None] for (traj_idx, t_idx) in zip(expert_cf_idxs, expert_cf_t_idxs)])

        '''
        sample from replay buffer data (1/2 batch size) (counterfacetuals)
        '''
        if self.for_rlpd:
            rb_cf_states = next(self.replay_buffer)
            import pdb; pdb.set_trace() # maybe need to change some shapes around
        else:
            rb_episode_trajs = [self.replay_buffer._sample_episode()[self.rb_buffer_obs_key][:-1] for _ in range(half_batch_size)]
            rb_cf_t_idxs = np.random.randint(self.horizon, size=(half_batch_size,))
            rb_cf_states = np.concatenate([rb_episode_trajs[idx][t][None] for idx, t in enumerate(rb_cf_t_idxs)])

        '''
        combine the counterfactals
        '''
        cf_states_np = np.concatenate([expert_cf_states, rb_cf_states], axis=0)
        if self.goal_is_metaworld_style:
            cf_states_np[:, -3:] = copy.deepcopy(expert_states_t_np[:, -3:])
        cf_states = torch.Tensor(cf_states_np).float().to(device)

        # not_counterfactual_logit = self.same_traj_classifier(expert_states_t)
        # counterfactual_logit = self.same_traj_classifier(cf_states)
        # traj_predictions = torch.cat([not_counterfactual_logit,  counterfactual_logit], dim=-1)
        # traj_labels = torch.cat([torch.ones_like(not_counterfactual_logit), torch.zeros_like(counterfactual_logit)], dim=-1)
        classify_states = torch.cat([expert_states_t, cf_states], dim=0)
        traj_labels = torch.cat([torch.ones((expert_states_t.size()[0], 1)), torch.zeros((cf_states.size()[0], 1))], dim=0).to(device)

        if self.train_classify_with_mixup:
            mixed_classify_states, traj_labels_a, traj_labels_b, lam = mixup_data(classify_states, traj_labels)
            mixed_traj_prediction_logits = self.same_traj_classifier(mixed_classify_states)
            loss_same_traj = mixup_criterion(
                self.bce_with_logits_criterion, mixed_traj_prediction_logits, traj_labels_a, traj_labels_b, lam
            )
        else:
            traj_prediction_logits = self.same_traj_classifier(classify_states)
            loss_same_traj = self.bce_with_logits_criterion(traj_prediction_logits, traj_labels)

        return {
            "ranking_loss": loss_monotonic,
            "same_traj_loss": loss_same_traj,
        }

    def _train_ranking_step(self, do_inference: bool = True):
        '''
        sample from expert data (factuals)

        do_inference option allows just sampling from the replay buffer which is useful
                     for training the classifier. this minimizes redundanct code
        '''
        expert_idxs = np.random.randint(self.num_expert_trajs, size=(self.batch_size,))
        expert_t_idxs = np.random.randint(self.horizon, size=(self.batch_size,))
        expert_other_t_idxs = np.random.randint(self.horizon, size=(self.batch_size,))
        labels = np.zeros((self.batch_size,))
        first_before = np.where(expert_t_idxs < expert_other_t_idxs)[0]
        labels[first_before] = 1.0 # idx is 1.0 if other timestep > timestep

        expert_states_t_np = np.concatenate([self.expert_data[traj_idx][0][t_idx][None] for (traj_idx, t_idx) in zip(expert_idxs, expert_t_idxs)])
        expert_states_other_t_np = np.concatenate([self.expert_data[traj_idx][0][t_idx][None] for (traj_idx, t_idx) in zip(expert_idxs, expert_other_t_idxs)])

        if self.add_state_noise:
            expert_states_t_np += np.random.normal(0, 0.01, size=expert_states_t_np.shape)
            expert_states_other_t_np += np.random.normal(0, 0.01, size=expert_states_other_t_np.shape)

        expert_states_t = torch.Tensor(expert_states_t_np).float().to(device)
        expert_states_other_t = torch.Tensor(expert_states_other_t_np).float().to(device)
        ranking_labels = F.one_hot(torch.Tensor(labels).long().to(device), 2).float()

        loss_monotonic = torch.Tensor([0.0])
        if do_inference:
            if not self.disable_ranking:
                if self.train_classify_with_mixup:
                    rank_states = torch.cat([expert_states_t, expert_states_other_t], dim=0)
                    rank_labels = torch.cat([ranking_labels[:, 0], ranking_labels[:, 1]], dim=0).unsqueeze(1)

                    mixed_rank_states, rank_labels_a, rank_labels_b, rank_lam = mixup_data(rank_states, rank_labels)
                    mixed_rank_prediction_logits = self.ranking_network(mixed_rank_states)
                    loss_monotonic = mixup_criterion(
                        self.bce_with_logits_criterion, mixed_rank_prediction_logits, rank_labels_a, rank_labels_b, rank_lam
                    )
                else:
                    expert_logits_t = self.ranking_network(expert_states_t)
                    expert_logits_other_t = self.ranking_network(expert_states_other_t)
                    expert_logits = torch.cat([expert_logits_t, expert_logits_other_t], dim=-1)

                    loss_monotonic = self.bce_with_logits_criterion(expert_logits, ranking_labels)

        return loss_monotonic, expert_states_t, expert_states_t_np

    def init_ranking(self):
        '''
        the ranking function is purely a function of expert data, so we can just train this once and not worry about
        training it again during RL
        '''
        if self.disable_ranking:
            print(f"LRF not configured to use a ranking function. Not training!")
            return


        ranking_init_losses_mean = []
        ranking_init_losses_std = []
        ranking_init_losses_steps = []
        running_losses = []
        window_size = 10
        num_init_steps = 5000
        print(f"Training the ranking function for {num_init_steps} steps:")
        self.ranking_network.train()
        for i in tqdm(range(num_init_steps)):
            self.ranking_optimizer.zero_grad()

            ranking_loss, _, _ = self._train_ranking_step()

            ranking_loss.backward()
            self.ranking_optimizer.step()

            running_losses.append(ranking_loss.item())
            if i % window_size == 0:
                ranking_init_losses_mean.append(np.mean(running_losses))
                ranking_init_losses_std.append(np.std(running_losses))
                ranking_init_losses_steps.append(i)
                running_losses = []

        ranking_init_losses_mean = np.array(ranking_init_losses_mean)
        ranking_init_losses_std = np.array(ranking_init_losses_std)

        # plot and save
        plt.clf(); plt.cla()
        plt.plot(ranking_init_losses_steps, ranking_init_losses_mean, label="ranking", color='blue')
        plt.fill_between(
            ranking_init_losses_steps,
            ranking_init_losses_mean - ranking_init_losses_std,
            ranking_init_losses_mean + ranking_init_losses_std,
            alpha=0.25,
            color='blue'
        )
        plt.legend()
        plt.savefig(f"{self.exp_dir}/ranking_init_loss.png")

        self.init_ranking_losses = ranking_init_losses_mean
        self.init_ranking_losses_std = ranking_init_losses_std
        self.init_ranking_steps = ranking_init_losses_steps
        print(f"Done training ranking function. Loss curve: {self.exp_dir}/ranking_init_loss.png")

        self.ranking_network.eval()

    def train(self, num_batches):
        for _ in tqdm(range(num_batches)):
            self.same_traj_optimizer.zero_grad()

            loss_dict = self._train_step()

            loss_dict["same_traj_loss"].backward()
            self.same_traj_optimizer.step()
            self.running_loss_same_traj.append(loss_dict["same_traj_loss"].item())

            self.train_step += 1

            if self.train_step % self.plot_and_save_frequency == 0:
                self.train_steps.append(self.train_step)
                self.losses_same_traj.append(np.mean(self.running_loss_same_traj))
                self.losses_std_same_traj.append(np.std(self.running_loss_same_traj))
                self.running_loss_same_traj = []
                self.plot_losses()
                self.save_models()

    def plot_losses(self):
        losses_same_traj = np.array(self.losses_same_traj)
        losses_std_same_traj = np.array(self.losses_std_same_traj)

        plt.clf(); plt.cla()
        plt.plot(self.train_steps, self.losses_same_traj, label="train", color='blue')
        plt.fill_between(
            self.train_steps,
            losses_same_traj - losses_std_same_traj,
            losses_same_traj + losses_std_same_traj,
            alpha=0.25,
            color='blue'
        )
        plt.legend()
        plt.savefig(f"{self.exp_dir}/training_loss_same_traj.png")

        # dump the raw data being plotted
        losses_dict = {
            "init_ranking_losses": self.init_ranking_losses,
            "init_ranking_losses_std": self.init_ranking_losses_std,
            "init_ranking_steps": self.init_ranking_steps,
            "train_iterations": self.train_steps,
            "loss_same_traj": self.losses_same_traj,
            "loss_same_traj_std": self.losses_std_same_traj,
        }
        pickle.dump(losses_dict, open(f"{self.exp_dir}/losses.pkl", "wb"))

    def save_models(self, save_dir: str = None):
        if save_dir is None:
            if not self.disable_ranking:
                torch.save(self.ranking_network.state_dict(), f"{self.exp_dir}/ranking_policy.pt")
            torch.save(self.same_traj_classifier.state_dict(), f"{self.exp_dir}/same_classifier_policy.pt")
        else:
            if not self.disable_ranking:
                torch.save(self.ranking_network.state_dict(), f"{save_dir}/ranking_policy.pt")
            torch.save(self.same_traj_classifier.state_dict(), f"{save_dir}/same_classifier_policy.pt")

    def load_models(self):
        print(f"loading models from disk in {self.exp_dir}")
        if not self.disable_ranking:
            self.ranking_network.load_state_dict(torch.load(f"{self.exp_dir}/ranking_policy.pt"))
        self.same_traj_classifier.load_state_dict(torch.load(f"{self.exp_dir}/same_classifier_policy.pt"))

    def eval_mode(self):
        self.same_traj_classifier.eval()

    def train_mode(self):
        self.same_traj_classifier.train()


class LearnedImageRewardFunction(LearnedRewardFunction):
    def __init__(self,
        obs_size: int,
        exp_dir: str,
        replay_buffer: ReplayBuffer,
        train_classify_with_mixup: bool = False,
        add_state_noise: bool = False,
        rb_buffer_obs_key: str = "observation",
        disable_classifier: bool = False, # ablation
        disable_ranking: bool = False, # GAIL / AIRL,
        goal_is_image: bool = False,
        train_classifier_with_goal_state_only: bool = False, # VICE,
        for_rlpd: bool = False,
        do_film_layer: bool = True,
    ):
        '''
        NOTE: assuming that someone else is responsible for making sure the replay buffer
              seeing fresh data on disk. we just like to have a pointer to it here to access
        '''
        self.exp_dir = exp_dir
        self.horizon = 100
        self.train_classify_with_mixup = train_classify_with_mixup
        self.add_state_noise = add_state_noise # let's call image shifts as state noise /shrug
        self.disable_ranking = disable_ranking
        self.disable_classifier = disable_classifier
        # self.aug = RandomShiftsAug(12)
        self.goal_is_image = goal_is_image
        self.train_classifier_with_goal_state_only = train_classifier_with_goal_state_only
        if self.train_classifier_with_goal_state_only:
            assert self.disable_ranking

        # training parameters
        self.batch_size = 96
        self.lr = 1e-4
        self.do_film_layer = do_film_layer
        if self.do_film_layer:
            self.weight_decay_rate = 1e-4
        else:
            self.weight_decay_rate = 0.0

        # network definitions
        if not self.disable_ranking:
            if self.goal_is_image:
                self.ranking_network = R3MImageGoalPolicy(freeze_backbone=True, film_layer_goal=self.do_film_layer)
            else:
                self.ranking_network = R3MPolicy(freeze_backbone=True, film_layer_goal=self.do_film_layer, state_only=for_rlpd)
            self.ranking_network.to(device)
            self.ranking_optimizer = optim.Adam(list(self.ranking_network.parameters()), lr=self.lr, weight_decay=self.weight_decay_rate)
        if self.goal_is_image:
            self.same_traj_classifier = R3MImageGoalPolicy(freeze_backbone=True, film_layer_goal=self.do_film_layer)
        else:
            self.same_traj_classifier = R3MPolicy(freeze_backbone=True, film_layer_goal=self.do_film_layer, state_only=for_rlpd)
        self.same_traj_classifier.to(device)
        self.same_traj_optimizer = optim.Adam(list(self.same_traj_classifier.parameters()), lr=self.lr, weight_decay=self.weight_decay_rate)
        self.bce_with_logits_criterion = torch.nn.BCEWithLogitsLoss()

        # make sure there is expert data
        self.expert_data_path = f"{self.exp_dir}/expert_data.hdf"
        self.expert_test_data_path = f"{self.exp_dir}/expert_test_data.hdf"
        assert os.path.exists(self.expert_data_path)
        self.expert_data_ptr = H5PyTrajDset(self.expert_data_path, read_only_if_exists=True)
        self.expert_data = [d for d in self.expert_data_ptr]
        self.expert_test_data_ptr = H5PyTrajDset(self.expert_test_data_path, read_only_if_exists=True)
        self.expert_test_data = [d for d in self.expert_test_data_ptr]
        self.num_expert_trajs = len(self.expert_data)
        assert self.num_expert_trajs == 100

        # replay buffer
        self.replay_buffer = replay_buffer
        self.rb_buffer_obs_key = rb_buffer_obs_key
        self.for_rlpd = for_rlpd

        # see if we need to do any upsampling
        self.resample_images = False
        if obs_size != (3, 224, 224):
            print(f"obs size: {obs_size} needs to be resized to (3, 224, 224)")
            self.resample_images = True
            self.resize = transforms.Resize(224)

        # bookkeeping
        self.train_step = 0
        self.plot_and_save_frequency = 100
        self.train_steps = []
        self.losses_same_traj = []
        self.losses_std_same_traj = []
        self.running_loss_same_traj = []

        self.seen_on_policy_data = False

        # init and bookkeeping
        self.init_ranking_steps = []
        self.init_ranking_losses = []
        self.init_ranking_losses_std = []
        self.init_ranking()


    def _preprocess_images(self, batch_images: torch.Tensor, eval: bool = False):
        # assumes the input is 0-255 as floats and of size (bs, c, h, w)
        batch = batch_images / 255.0

        if self.resample_images:
            batch = self.resize(batch)

        # if self.add_state_noise and not eval:
        #     batch = self.aug(batch)

        return batch

    def _calculate_reward(self, obs: torch.Tensor, goal: torch.Tensor=None, airl_style_reward: bool = False, take_log_reward: bool = False, take_d_ratio: bool = False, lgn_multiplier=1.0, eps=1e-6):
        '''
        this code path is executed by the drqv2 agent updating the stale rewards
        '''
        batch_imgs = obs
        batch_goals = goal

        with torch.no_grad():
            self.eval_mode()

            batch_imgs = self._preprocess_images(batch_imgs, eval=True)
            if self.goal_is_image:
                batch_goals = self._preprocess_images(batch_goals, eval=True)

            if self.disable_ranking:
                mask = torch.sigmoid(self.same_traj_classifier(batch_imgs, batch_goals)).cpu().numpy()
                reward = mask
            elif not self.seen_on_policy_data or self.disable_classifier:
                progress = torch.sigmoid(self.ranking_network(batch_imgs, batch_goals)).cpu().numpy()
                reward = progress
            else:
                mask = torch.sigmoid(self.same_traj_classifier(batch_imgs, batch_goals)).cpu().numpy()
                progress = torch.sigmoid(self.ranking_network(batch_imgs, batch_goals)).cpu().numpy()
                reward = progress * mask

            self.train_mode()

        if airl_style_reward:
            reward_clamped = np.clip(reward, self.eps, 1 - self.eps) # numerical stability
            reward = np.log(reward_clamped) - np.log(1 - reward_clamped)
        elif take_log_reward:
            if self.disable_classifier:
                progress_clamped = np.clip(progress, eps, 1 - eps) # numerical stability
                reward = np.log(progress_clamped)
            else:
                mask_clamped = np.clip(mask, eps, 1 - eps) # numerical stability
                progress_clamped = np.clip(progress, eps, 1 - eps) # numerical stability
                if take_d_ratio:
                    reward = np.log(progress_clamped) + lgn_multiplier * (np.log(mask_clamped) - np.log(1 - mask_clamped))
                else:
                    reward = np.log(progress_clamped) + lgn_multiplier * np.log(mask_clamped)

        return reward

    def _train_ranking_step(self, do_inference: bool = True):
        '''
        sample from expert data (factuals)

        do_inference option allows just sampling from the replay buffer which is useful
                     for training the classifier. this minimizes redundanct code
        '''

        '''
        sample from expert data (factuals)
        '''
        expert_idxs = np.random.randint(self.num_expert_trajs, size=(self.batch_size,))
        expert_t_idxs = np.random.randint(self.horizon, size=(self.batch_size,))
        expert_other_t_idxs = np.random.randint(self.horizon, size=(self.batch_size,))
        labels = np.zeros((self.batch_size,))
        first_before = np.where(expert_t_idxs < expert_other_t_idxs)[0]
        labels[first_before] = 1.0 # idx is 1.0 if other timestep > timestep

        expert_states_t_np = np.concatenate([self.expert_data[traj_idx][0][t_idx][None] for (traj_idx, t_idx) in zip(expert_idxs, expert_t_idxs)])
        expert_states_other_t_np = np.concatenate([self.expert_data[traj_idx][0][t_idx][None] for (traj_idx, t_idx) in zip(expert_idxs, expert_other_t_idxs)])
        expert_goals_np = np.concatenate([self.expert_data[traj_idx][3][t_idx][None] for (traj_idx, t_idx) in zip(expert_idxs, expert_t_idxs)])

        expert_states_t = torch.Tensor(expert_states_t_np).float().to(device)
        expert_states_other_t = torch.Tensor(expert_states_other_t_np).float().to(device)
        expert_goals = torch.Tensor(expert_goals_np).float().to(device)

        expert_states_p_t = self._preprocess_images(expert_states_t)
        expert_states_other_p_t = self._preprocess_images(expert_states_other_t)
        if self.goal_is_image:
            expert_goals = self._preprocess_images(expert_goals)

        ranking_labels = F.one_hot(torch.Tensor(labels).long().to(device), 2).float()

        loss_monotonic = torch.Tensor([0.0])
        if do_inference:
            if not self.disable_ranking:
                if self.train_classify_with_mixup:
                    rank_states = torch.cat([expert_states_p_t, expert_states_other_p_t], dim=0)
                    rank_goals = torch.cat([expert_goals, expert_goals], dim=0)
                    rank_labels = torch.cat([ranking_labels[:, 0], ranking_labels[:, 1]], dim=0).unsqueeze(1)

                    mixed_rank_states, rank_labels_a, rank_labels_b, rank_lam, mixed_rank_goals = mixup_data(rank_states, rank_labels, goals=rank_goals)
                    mixed_rank_prediction_logits = self.ranking_network(mixed_rank_states, mixed_rank_goals)
                    loss_monotonic = mixup_criterion(
                        self.bce_with_logits_criterion, mixed_rank_prediction_logits, rank_labels_a, rank_labels_b, rank_lam
                    )
                else:
                    expert_logits_t = self.ranking_network(expert_states_p_t, expert_goals)
                    expert_logits_other_t = self.ranking_network(expert_states_other_p_t, expert_goals)
                    expert_logits = torch.cat([expert_logits_t, expert_logits_other_t], dim=-1)

                    loss_monotonic = self.bce_with_logits_criterion(expert_logits, ranking_labels)

        return loss_monotonic, expert_states_p_t, expert_goals


    def _train_step(self):
        '''
        this will train the classifier for a full step, using the ranking steps code to sample from the expert data
        '''
        self.seen_on_policy_data = True

        '''
        sample from expert data (factuals)
        '''
        loss_monotonic, expert_states_p_t, expert_goals = self._train_ranking_step(do_inference=False)

        '''
        sample another set (1/2 batch size) from expert data (counterfactuals)
        '''
        half_batch_size = self.batch_size // 2
        expert_cf_idxs = np.random.randint(self.num_expert_trajs, size=(half_batch_size,))
        if self.train_classifier_with_goal_state_only:
            expert_cf_t_idxs = np.ones(shape=(half_batch_size,), dtype=np.int64) * (self.horizon - 1)
        else:
            expert_cf_t_idxs = np.random.randint(self.horizon, size=(half_batch_size,))
        expert_cf_states = np.concatenate([self.expert_data[traj_idx][0][t_idx][None] for (traj_idx, t_idx) in zip(expert_cf_idxs, expert_cf_t_idxs)])

        '''
        sample from replay buffer data (1/2 batch size) (counterfactuals)
        '''
        if self.for_rlpd:
            next_batch = next(self.replay_buffer)
            rb_cf_states = next_batch['observations']['pixels'] # bs, h, w, c, frame_stack
            rb_cf_states = np.transpose(np.squeeze(rb_cf_states), (0, 3, 1, 2))
        else:
            rb_episode_trajs = [self.replay_buffer._sample_episode()[self.rb_buffer_obs_key][:-1] for _ in range(half_batch_size)]
            rb_cf_t_idxs = np.random.randint(self.horizon, size=(half_batch_size,))
            rb_cf_states = np.concatenate([rb_episode_trajs[idx][t][None] for idx, t in enumerate(rb_cf_t_idxs)])

        '''
        combine the counterfactals
        '''
        cf_states_np = np.concatenate([expert_cf_states, rb_cf_states], axis=0)
        cf_states = torch.Tensor(cf_states_np).float().to(device)
        cf_states_p = self._preprocess_images(cf_states)

        # not_counterfactual_logit = self.same_traj_classifier(expert_states_t)
        # counterfactual_logit = self.same_traj_classifier(cf_states)
        # traj_predictions = torch.cat([not_counterfactual_logit,  counterfactual_logit], dim=-1)
        # traj_labels = torch.cat([torch.ones_like(not_counterfactual_logit), torch.zeros_like(counterfactual_logit)], dim=-1)
        classify_states = torch.cat([expert_states_p_t, cf_states_p], dim=0)
        classify_goals = torch.cat([expert_goals, expert_goals], dim=0)
        traj_labels = torch.cat([torch.ones((expert_states_p_t.size()[0], 1)), torch.zeros((cf_states_p.size()[0], 1))], dim=0).to(device)

        if self.train_classify_with_mixup:
            mixed_classify_states, traj_labels_a, traj_labels_b, lam, mixed_goals = mixup_data(classify_states, traj_labels, goals=classify_goals)
            mixed_traj_prediction_logits = self.same_traj_classifier(mixed_classify_states, mixed_goals)
            loss_same_traj = mixup_criterion(
                self.bce_with_logits_criterion, mixed_traj_prediction_logits, traj_labels_a, traj_labels_b, lam
            )
        else:
            traj_prediction_logits = self.same_traj_classifier(classify_states, classify_goals)
            loss_same_traj = self.bce_with_logits_criterion(traj_prediction_logits, traj_labels)

        return {
            "ranking_loss": loss_monotonic,
            "same_traj_loss": loss_same_traj,
        }


if __name__ == "__main__":
    from policy_learning.envs import ImageMetaworldEnv

    envs = [
        "assembly", "drawer-open", "hammer", "door-close", "push",
        "reach", "button-press-topdown", "door-open"
    ]

    for env_str in tqdm(envs):
        env = ImageMetaworldEnv(env_str, camera_name="left_cap2", high_res_env=False)
        expert_data_dir = os.path.expanduser(f"~/code/rewardlearning-vid/ROT/ROT/expert_demos/{env_str}")
        if not os.path.exists(expert_data_dir):
            os.makedirs(expert_data_dir)
        check_and_generate_expert_data(expert_data_dir, env_str, env, 100, False, False)
