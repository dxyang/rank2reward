import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import sys
import pickle

from pathlib import Path

from dm_env import StepType, specs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

import drqv2.dmc as dmc

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

from reward_extraction.metaworld_utils import process_obs, random_reset, VALID_METAWORLD_ENVS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LowDimMetaworldEnv():
    def __init__(self, env_str):
        self._env_str = env_str
        assert self._env_str in VALID_METAWORLD_ENVS

        reach_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{self._env_str}-v2-goal-observable"]

        self._env = reach_goal_observable_cls()
        self._env.random_init = False
        self._env._freeze_rand_vec = False
        self._env._obs_dict_state_space = True
        self._env._do_render_for_obs = False      # maybe shouldn't always render?
        self._env._render_higher_res_obs = False  # maybe True?
        self._env.max_path_length = 100           # is this a good choice?

        self._condition_on_initial_state = True   # we're pretty much always doing this?

        self.last_received_reward_changed = True
        self.last_received_reward = 0

        self.horizon = self._env.max_path_length
        self.step_t = 0

    def _convert_obs_to_timestep(self, obs_dict, step_type, action=None, reward=0.0, info={}):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        obs = process_obs(obs_dict["state_observation"], self._env_str, initial_state=self._initial_state) # state, goal

        return dmc.ExtendedTimeStep(observation=obs,
                                    step_type=step_type,
                                    action=action,
                                    reward=reward,
                                    success=info['success'] if step_type is not StepType.FIRST else float(False),
                                    og_reward=reward,
                                    metaworld_state_obs=obs_dict["state_observation"].astype(np.float32),
                                    discount=1.0)

    def reset(self, goal_pos=None, hand_init=None, obj_pos=None):
        obs_dict, obj_pos, goal_pos = random_reset(self._env_str, self._env, goal_pos=goal_pos, hand_init=hand_init, obj_pos=obj_pos)

        # cache the initial state so we can include it into the state space
        self._initial_state = process_obs(obs_dict["state_observation"], self._env_str, no_goal=True).copy()

        self.step_t = 0

        return self._convert_obs_to_timestep(obs_dict, StepType.FIRST)

    def step(self, action):
        obs_dict, reward, done, info = self._env.step(action)
        self.step_t += 1

        self.last_received_reward = reward
        self.last_received_reward_changed = True

        if done:
            return self._convert_obs_to_timestep(obs_dict, StepType.LAST, action, reward, info)
        else:
            return self._convert_obs_to_timestep(obs_dict, StepType.MID, action, reward, info)

    def get_last_received_reward(self):
        assert self.last_received_reward_changed
        self.last_received_reward_changed = False
        return self.last_received_reward

    def observation_spec(self):
        if self._env_str == "reach":
            obs_shape = (6,)
            if self._condition_on_initial_state:
                obs_shape = (9,)
        elif self._env_str in [
            "push", "door-open", "door-close",
            "assembly", "bin-picking", "button-press-topdown",
            "drawer-open", "hammer"
        ]:
            obs_shape = (10,)
            if self._condition_on_initial_state:
                obs_shape = (17,)
        else:
            assert False

        env_obs_spec = specs.BoundedArray(shape=obs_shape,
                                            dtype=np.float32,
                                            minimum=-np.inf,
                                            maximum=np.inf,
                                            name='observation')
        return env_obs_spec

    def action_spec(self):
        env_action_spec = specs.BoundedArray(shape=(4,),
                                            dtype=np.float32,
                                            minimum=-1,
                                            maximum=1,
                                            name='action')
        return env_action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)

class LowDimOnlineCustomRewardMetaworldEnv(LowDimMetaworldEnv):
    '''
    do the same thing as LowDimMetaworldEnv but override the reward
    '''
    def __init__(self, env_str, lrf, airl_style_reward: bool = False, take_log_reward: bool = False, take_d_ratio: bool = False, lgn_multiplier: float = 1.0, eps: float=1e-5):
        super().__init__(env_str)
        self.learned_reward_function = lrf
        self.airl_style_reward = airl_style_reward
        self.take_log_reward = take_log_reward
        self.take_d_ratio = take_d_ratio
        self.lgn_multiplier = lgn_multiplier
        self.eps = eps

    def _convert_obs_to_timestep(self, obs_dict, step_type, action=None, reward=0.0, info={}):
        og_reward = reward

        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        obs = process_obs(obs_dict["state_observation"], self._env_str, initial_state=self._initial_state)

        if step_type is not StepType.FIRST:
            with torch.no_grad():
                self.learned_reward_function.eval_mode()

                batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0)).float().to(device)

                if self.learned_reward_function.disable_ranking:
                    same_traj_val = torch.sigmoid(self.learned_reward_function.same_traj_classifier(batch_obs)).cpu().item()
                    reward = same_traj_val
                elif self.learned_reward_function.disable_classifier:
                    ranking_val = torch.sigmoid(self.learned_reward_function.ranking_network(batch_obs)).cpu().item()
                    reward = ranking_val
                else:
                    same_traj_val = torch.sigmoid(self.learned_reward_function.same_traj_classifier(batch_obs)).cpu().item()
                    ranking_val = torch.sigmoid(self.learned_reward_function.ranking_network(batch_obs)).cpu().item()
                    reward = ranking_val * same_traj_val

                if self.airl_style_reward:
                    reward_clamped = np.clip(reward, self.eps, 1 - self.eps) # numerical stability
                    reward = np.log(reward_clamped) - np.log(1 - reward_clamped)
                elif self.take_log_reward:
                    if self.learned_reward_function.disable_classifier:
                        ranking_val_clamped = np.clip(ranking_val, self.eps, 1 - self.eps) # numerical stability
                        reward = np.log(ranking_val_clamped)
                    else:
                        same_traj_val_clamped = np.clip(same_traj_val, self.eps, 1 - self.eps) # numerical stability
                        ranking_val_clamped = np.clip(ranking_val, self.eps, 1 - self.eps) # numerical stability
                        if self.take_d_ratio:
                            reward = np.log(ranking_val_clamped) + self.lgn_multiplier * (np.log(same_traj_val_clamped) - np.log(1 - same_traj_val_clamped))
                        else:
                            reward = np.log(ranking_val_clamped) + self.lgn_multiplier * np.log(same_traj_val_clamped)

                self.learned_reward_function.train_mode()

        return dmc.ExtendedTimeStep(observation=obs,
                                    step_type=step_type,
                                    action=action,
                                    reward=reward,
                                    success=info['success'] if step_type is not StepType.FIRST else float(False),
                                    og_reward=og_reward,
                                    metaworld_state_obs=obs_dict["state_observation"].astype(np.float32),
                                    discount=1.0)

class LowDimStateCustomRewardFromImagesMetaworldEnv(LowDimMetaworldEnv):
    '''
    do the same thing as LowDimMetaworldEnv but override the reward (calculated from images instead)
    '''
    def __init__(self, env_str: str, camera_name: str, high_res_env: bool, lrf):
        super().__init__(env_str)
        self._env._do_render_for_obs = True
        self._env._render_higher_res_obs = high_res_env
        self._env._camera_name = camera_name

        self.learned_reward_function = lrf
        self.resize_to_resnet = transforms.Resize(224)

    def _convert_obs_to_timestep(self, obs_dict, step_type, action=None, reward=0.0, info={}):
        og_reward = reward

        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        state_obs = process_obs(obs_dict["state_observation"], self._env_str, initial_state=self._initial_state)

        img_obs = np.transpose(obs_dict["image_observation"], (2, 0, 1)) # CHW image, 0-255 int range, for storage!
        goal = obs_dict["state_observation"][-3:]

        if step_type is not StepType.FIRST:
            with torch.no_grad():
                self.learned_reward_function.eval_mode()

                batch_obs = torch.from_numpy(np.expand_dims(img_obs, axis=0)).float().to(device) / 255.0
                if not self._env._render_higher_res_obs:
                    batch_obs = self.resize_to_resnet(batch_obs)
                batch_goal = torch.from_numpy(np.expand_dims(goal, axis=0)).float().to(device)

                same_traj_val = torch.sigmoid(self.learned_reward_function.same_traj_classifier(batch_obs, batch_goal)).cpu().item()
                if self.learned_reward_function.disable_ranking:
                    reward = same_traj_val
                else:
                    ranking_val = torch.sigmoid(self.learned_reward_function.ranking_network(batch_obs, batch_goal)).cpu().item()
                    reward = ranking_val * same_traj_val

                self.learned_reward_function.train_mode()

        return dmc.ImageExtendedTimeStep(observation=state_obs,
                                         step_type=step_type,
                                         action=action,
                                         reward=reward,
                                         success=info['success'] if step_type is not StepType.FIRST else float(False),
                                         og_reward=og_reward,
                                         metaworld_state_obs=obs_dict["state_observation"].astype(np.float32),
                                         discount=1.0,
                                         image=img_obs)

class ImageMetaworldEnv(LowDimMetaworldEnv):
    '''
    do the same thing as LowDimMetaworldEnv but state space is images
    '''
    def __init__(self, env_str: str, camera_name: str, high_res_env: bool, rlpd_res: bool = False):
        super().__init__(env_str)
        self._env._do_render_for_obs = True
        self._env._render_higher_res_obs = high_res_env
        self._env._camera_name = camera_name
        self._env._render_rlpd_res_obs = rlpd_res

    def _convert_obs_to_timestep(self, obs_dict, step_type, action=None, reward=0.0, info={}):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        obs = np.transpose(obs_dict["image_observation"], (2, 0, 1)) # CHW image
        goal = obs_dict["state_desired_goal"]

        return dmc.ExtendedTimeStep(observation=obs,
                                    step_type=step_type,
                                    action=action,
                                    reward=reward,
                                    success=info['success'] if step_type is not StepType.FIRST else float(False),
                                    og_reward=reward,
                                    metaworld_state_obs=obs_dict["state_observation"].astype(np.float32),
                                    discount=1.0)

    def observation_spec(self):
        if self._env._render_higher_res_obs:
            pixels_shape = (224, 224, 3)
        elif self._env._render_rlpd_res_obs:
            pixels_shape = (64, 64, 3)
        else:
            pixels_shape = (84, 84, 3)
        num_frames = 1
        env_obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')
        return env_obs_spec

class ImageOnlineCustomRewardMetaworldEnv(ImageMetaworldEnv):
    '''
    do the same thing as ImageMetaworldEnv but override the reward
    '''
    def __init__(self, env_str: str, camera_name: str, high_res_env: bool, lrf, airl_style_reward: bool = False, rlpd_res: bool = False, take_log_reward: bool = False, take_d_ratio: bool = False, lgn_multiplier: float = 1.0, eps: float=1e-5):
        super().__init__(env_str, camera_name, high_res_env, rlpd_res)
        self.learned_reward_function = lrf
        self.resize_to_resnet = transforms.Resize(224)
        self.airl_style_reward = airl_style_reward
        self.take_log_reward = take_log_reward
        self.take_d_ratio = take_d_ratio
        self.lgn_multiplier = lgn_multiplier
        self.eps = eps

    def _convert_obs_to_timestep(self, obs_dict, step_type, action=None, reward=0.0, info={}):
        og_reward = reward

        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        obs = np.transpose(obs_dict["image_observation"], (2, 0, 1)) # CHW image, 0-255 int range, for storage!
        goal = obs_dict["state_observation"][-3:]

        if step_type is not StepType.FIRST:
            with torch.no_grad():
                self.learned_reward_function.eval_mode()

                batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0)).float().to(device) / 255.0
                if not self._env._render_higher_res_obs:
                    batch_obs = self.resize_to_resnet(batch_obs)
                batch_goal = torch.from_numpy(np.expand_dims(goal, axis=0)).float().to(device)

                if self.learned_reward_function.disable_ranking:
                    same_traj_val = torch.sigmoid(self.learned_reward_function.same_traj_classifier(batch_obs, batch_goal)).cpu().item()
                    reward = same_traj_val
                elif self.learned_reward_function.disable_classifier:
                    ranking_val = torch.sigmoid(self.learned_reward_function.ranking_network(batch_obs, batch_goal)).cpu().item()
                    reward = ranking_val
                else:
                    same_traj_val = torch.sigmoid(self.learned_reward_function.same_traj_classifier(batch_obs, batch_goal)).cpu().item()
                    ranking_val = torch.sigmoid(self.learned_reward_function.ranking_network(batch_obs, batch_goal)).cpu().item()
                    reward = ranking_val * same_traj_val

                if self.airl_style_reward:
                    reward_clamped = np.clip(reward, self.eps, 1 - self.eps) # numerical stability
                    reward = np.log(reward_clamped) - np.log(1 - reward_clamped)
                elif self.take_log_reward:
                    if self.learned_reward_function.disable_classifier:
                        ranking_val_clamped = np.clip(ranking_val, self.eps, 1 - self.eps) # numerical stability
                        reward = np.log(ranking_val_clamped)
                    else:
                        same_traj_val_clamped = np.clip(same_traj_val, self.eps, 1 - self.eps) # numerical stability
                        ranking_val_clamped = np.clip(ranking_val, self.eps, 1 - self.eps) # numerical stability
                        if self.take_d_ratio:
                            reward = np.log(ranking_val_clamped) + self.lgn_multiplier * (np.log(same_traj_val_clamped) - np.log(1 - same_traj_val_clamped))
                        else:
                            reward = np.log(ranking_val_clamped) + self.lgn_multiplier * np.log(same_traj_val_clamped)


                self.learned_reward_function.train_mode()

        return dmc.ExtendedTimeStep(observation=obs,
                                    step_type=step_type,
                                    action=action,
                                    reward=reward,
                                    success=info['success'] if step_type is not StepType.FIRST else float(False),
                                    og_reward=og_reward,
                                    metaworld_state_obs=obs_dict["state_observation"].astype(np.float32),
                                    discount=1.0)

class TcnImageMetaworldEnv(ImageMetaworldEnv):
    '''
    do the same thing as ImageMetaworldEnv but override the reward
    '''
    def __init__(self, env_str: str, camera_name: str, high_res_env: bool, expert_data_path: str, tcn_model_path: str, rlpd_res: bool = False):
        from tcn.tcn import TCNModel
        from reward_extraction.data import H5PyTrajDset

        super().__init__(env_str, camera_name, high_res_env, rlpd_res)
        self.resize_to_inception = transforms.Resize(299)

        # load the pretrained TCN model
        self.tcn_model = TCNModel()
        self.tcn_model.load_state_dict(torch.load(tcn_model_path))
        self.tcn_model.eval()
        self.tcn_model.to(device)
        self.tcn_alpha = 1.0 # described in paper as empirically chosen
        self.tcn_beta = 1.0 # described in paper as empirically chosen
        self.tcn_gamma = 0.1 # described in paper as a small constant

        # TCN preprocess
        self.TCN_IMAGE_SIZE = (299,299)
        self.resize = transforms.Resize(self.TCN_IMAGE_SIZE)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # load the expert data
        self.expert_data_path = expert_data_path
        self.expert_data_ptr = H5PyTrajDset(self.expert_data_path, read_only_if_exists=True)
        self.expert_obs_traj = self.expert_data_ptr[0][0]

    def _convert_obs_to_timestep(self, obs_dict, step_type, action=None, reward=0.0, info={}):
        og_reward = reward

        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        obs = np.transpose(obs_dict["image_observation"], (2, 0, 1)) # CHW image, 0-255 int range, for storage!
        goal = obs_dict["state_observation"][-3:]

        if step_type is not StepType.FIRST:
            # find matching observation in expert trajectory (v_t)
            v_t = torch.from_numpy(np.expand_dims(self.expert_obs_traj[self.step_t], axis=0)).float().to(device) / 255.0

            # calculate embeddings (v_t, w_t)
            w_t = torch.from_numpy(np.expand_dims(obs, axis=0)).float().to(device) / 255.0

            pp_v_t = self.normalize(self.resize(v_t))
            pp_w_t = self.normalize(self.resize(w_t))

            with torch.no_grad():
                embedding_v_t = self.tcn_model(pp_v_t)
                embedding_w_t = self.tcn_model(pp_w_t)

                # reward is huber-style loss
                dist = torch.pow(torch.abs(embedding_v_t - embedding_w_t), 2).sum(dim=1)
                reward = -self.tcn_alpha * dist - self.tcn_beta * torch.sqrt(self.tcn_gamma + dist)
                reward = reward.item()

        return dmc.ExtendedTimeStep(observation=obs,
                                    step_type=step_type,
                                    action=action,
                                    reward=reward,
                                    success=info['success'] if step_type is not StepType.FIRST else float(False),
                                    og_reward=og_reward,
                                    metaworld_state_obs=obs_dict["state_observation"].astype(np.float32),
                                    discount=1.0)


if __name__ == '__main__':
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    from reward_extraction.metaworld_experts import get_expert_policy
    from reward_extraction.reward_functions import generate_expert_data

    # wrap env and just iterate a bunch of times and run the expert policy
    envs = [
        "assembly", "drawer-open", "hammer", "door-close", "push",
        "reach", "bin-picking", "button-press-topdown", "door-open"
    ]

    # env_str = "push"
    for env_str in envs:
        res_dir = f"all_env_figure/{env_str}"
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        expert_data_path = f"{res_dir}/trash"
        env = LowDimMetaworldEnv(env_str)
        # env = ImageMetaworldEnv(env_str, camera_name="right_cap2")
        expert_policy = get_expert_policy(env_str)
        generate_expert_data(env_str, env, expert_data_path=expert_data_path, num_trajs=1, render=False, generate_videos=True)

    # for i in range(10):
    #     time_step = env.reset()

    #     step = 0
    #     rewards = []

    #     while not time_step.last():
    #         env.render()
    #         action = expert_policy.get_action(time_step.metaworld_state_obs)
    #         time_step = env.step(action)
    #         rewards.append(time_step.reward)
    #         step += 1

    #     total_reward = np.sum(rewards)
    #     print(total_reward)


