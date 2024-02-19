import copy
import functools
import glob
import os
import pickle
from re import I
from typing import List
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from tqdm import tqdm

from drqv2.replay_buffer import ReplayBuffer
from drqv2.video import TrainVideoRecorder

from reward_extraction.data import H5PyTrajDset
from reward_extraction.metaworld_experts import get_expert_policy
from reward_extraction.models import Policy, R3MPolicy
from policy_learning.envs import ImageMetaworldEnv, LowDimMetaworldEnv
from reward_extraction.metaworld_utils import set_reward_plot_limits

from reward_extraction.reward_functions import LearnedRewardFunction, LearnedImageRewardFunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
generating plots on the offline expert data
'''
def generate_all_traj_rankings(lrf: LearnedRewardFunction, analysis_dir: Path, is_input_images: bool, on_train_data: bool, generate_videos: bool = False):
    '''
    plot every trajectory from the expert data ranking output
    expect to see masks ranking everything high, progress increasing over each trajectory,
    and ranking product to be pretty much the same as ranking
    '''

    plot_dir = analysis_dir / "all_traj_rankings"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    with torch.no_grad():
        trajs = copy.deepcopy(lrf.expert_data) if on_train_data else copy.deepcopy(lrf.expert_test_data)
        split_str = "train" if on_train_data else "test"

        traj_progresses = []
        traj_masks = []
        traj_rankings = []

        if is_input_images:
            img_shape = trajs[0][0].shape[1:] # CHW
            render_size = img_shape[1]
            if render_size % 16 != 0:
                render_size = (render_size // 16 + 1) * 16

        for traj_idx, traj in enumerate(trajs):
            if is_input_images:
                states_np = traj[0][:-1]
                goals_np = traj[3][:-1]

                states_0255int = torch.Tensor(states_np).float().to(device)
                goals = torch.Tensor(goals_np).float().to(device)
                states = lrf._preprocess_images(states_0255int)
                if lrf.goal_is_image:
                    goals = lrf._preprocess_images(goals)

                mask = torch.sigmoid(lrf.same_traj_classifier(states, goals)).cpu().numpy()
                if lrf.disable_ranking:
                    progress = np.ones_like(mask)
                else:
                    progress = torch.sigmoid(lrf.ranking_network(states, goals)).cpu().numpy()
                ranking = progress * mask

                if generate_videos:
                    video_recorder = TrainVideoRecorder(plot_dir, folder_name=f"{'train' if on_train_data else 'test'}_videos", render_size=render_size)
                    for frame_idx, img in enumerate(states_np):
                        img = np.transpose(img, (1, 2, 0))
                        if frame_idx == 0:
                            video_recorder.init(img)
                        else:
                            video_recorder.record(img)
                    video_recorder.save(f"{str(traj_idx).zfill(3)}.mp4")
            else:
                states_np = traj[0][:-1]
                states_tensor = torch.Tensor(states_np).float().to(device)

                mask = torch.sigmoid(lrf.same_traj_classifier(states_tensor)).cpu().numpy()
                if lrf.disable_ranking:
                    progress = np.ones_like(mask)
                else:
                    progress = torch.sigmoid(lrf.ranking_network(states_tensor)).cpu().numpy()
                ranking = progress * mask

            traj_progresses.append(progress)
            traj_masks.append(mask)
            traj_rankings.append(ranking)

        plt.clf(); plt.cla()
        for p in traj_progresses:
            plt.plot(p)
        plt.ylim(0, 1)
        plt.savefig(f"{plot_dir}/{split_str}_alltrajs_progress.png")

        plt.clf(); plt.cla()
        for p in traj_masks:
            plt.plot(p)
        plt.ylim(0, 1)
        plt.savefig(f"{plot_dir}/{split_str}_alltrajs_mask.png")

        plt.clf(); plt.cla()
        for p in traj_rankings:
            plt.plot(p)
        plt.ylim(0, 1)
        plt.savefig(f"{plot_dir}/{split_str}_alltrajs_ranking.png")

def generate_counterfactual_plots(lrf: LearnedRewardFunction, analysis_dir: Path, is_input_images: bool, on_train_data: bool):
    '''
    take the goals from 10 trajectories and plot the other trajectories with their goals counter
    factually swapped. we would expect the classifier to recognize these are ood and push them down
    '''

    plot_dir = analysis_dir / "counterfactuals"
    mask_dir = plot_dir / "mask"
    progress_dir = plot_dir / "progress"
    ranking_dir = plot_dir / "ranking"
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
        os.makedirs(progress_dir)
        os.makedirs(ranking_dir)

    num_cf_plots = 10

    with torch.no_grad():
        trajs = copy.deepcopy(lrf.expert_data) if on_train_data else copy.deepcopy(lrf.expert_test_data)
        split_str = "train" if on_train_data else "test"

        for idx in range(num_cf_plots):
            idx_str = str(idx).zfill(3)
            traj_progresses = []
            traj_masks = []
            traj_rankings = []

            cf_goal = trajs[idx][3][:-1]
            cf_goals = torch.Tensor(cf_goal).float().to(device)
            if is_input_images:
                if lrf.goal_is_image:
                    cf_goals = lrf._preprocess_images(cf_goals)

            factual_progress, factual_mask, factual_ranking = 0,0,0

            for traj_idx, traj in enumerate(trajs):
                if is_input_images:
                    states_np = traj[0][:-1]

                    states_0255int = torch.Tensor(states_np).float().to(device)
                    states = lrf._preprocess_images(states_0255int)

                    mask = torch.sigmoid(lrf.same_traj_classifier(states, cf_goals)).cpu().numpy()
                    if lrf.disable_ranking:
                        progress = np.ones_like(mask)
                    else:
                        progress = torch.sigmoid(lrf.ranking_network(states, cf_goals)).cpu().numpy()
                    ranking = progress * mask

                else:
                    states_np = traj[0][:-1]
                    if lrf.goal_is_metaworld_style:
                        states_np[:, -3:] = cf_goal
                    states_tensor = torch.Tensor(states_np).float().to(device)

                    mask = torch.sigmoid(lrf.same_traj_classifier(states_tensor)).cpu().numpy()
                    if lrf.disable_ranking:
                        progress = np.ones_like(mask)
                    else:
                        progress = torch.sigmoid(lrf.ranking_network(states_tensor)).cpu().numpy()
                    ranking = progress * mask

                if traj_idx == idx:
                    factual_progress = progress
                    factual_mask = mask
                    factual_ranking = ranking
                else:
                    traj_progresses.append(progress)
                    traj_masks.append(mask)
                    traj_rankings.append(ranking)

            plt.clf(); plt.cla()
            for p in traj_progresses:
                plt.plot(p, c='r')
            plt.plot(factual_progress, c='b')
            plt.ylim(0, 1)
            plt.savefig(f"{progress_dir}/{split_str}_{idx_str}_cftrajs_progress.png")

            plt.clf(); plt.cla()
            for p in traj_masks:
                plt.plot(p, c='r')
            plt.plot(factual_mask, c='b')
            plt.ylim(0, 1)
            plt.savefig(f"{mask_dir}/{split_str}_{idx_str}_cftrajs_mask.png")

            plt.clf(); plt.cla()
            for p in traj_rankings:
                plt.plot(p, c='r')
            plt.plot(factual_ranking, c='b')
            plt.ylim(0, 1)
            plt.savefig(f"{ranking_dir}/{split_str}_{idx_str}_cftrajs_ranking.png")


def generate_counterfactual_reward_at_diff_states_plots(env_str: str, lrf: LearnedRewardFunction, analysis_dir: Path, is_input_images: bool, on_train_data: bool):
    '''
    take the goals from 10 trajectories and plot the other trajectories with their goals counter
    factually swapped. plot as a function of hand xy instead of over the trajectory
    '''
    # plotting settings
    alpha = 1.0
    linewidths=0.2
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)

    # dir bookkeeping
    plot_dir = analysis_dir / "xy_cf_plots"
    mask_dir = plot_dir / "mask"
    progress_dir = plot_dir / "progress"
    ranking_dir = plot_dir / "ranking"
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
        os.makedirs(progress_dir)
        os.makedirs(ranking_dir)

    num_cf_plots = 10

    with torch.no_grad():
        trajs = copy.deepcopy(lrf.expert_data) if on_train_data else copy.deepcopy(lrf.expert_test_data)
        split_str = "train" if on_train_data else "test"

        for idx in range(num_cf_plots):
            idx_str = str(idx).zfill(3)

            traj_lowdimobs = []
            traj_progresses = []
            traj_masks = []
            traj_rankings = []

            cf_goal = trajs[idx][3][:-1]
            cf_goals = torch.Tensor(cf_goal).float().to(device)
            if lrf.goal_is_image:
                cf_goals = lrf._preprocess_images(cf_goals)

            factual_progress, factual_mask, factual_ranking, factual_lowdimobs = 0,0,0, 0

            for traj_idx, traj in enumerate(trajs):
                full_states_np = traj[4][:-1]

                if is_input_images:
                    states_np = traj[0][:-1]

                    states_0255int = torch.Tensor(states_np).float().to(device)
                    states = lrf._preprocess_images(states_0255int)
                    mask = torch.sigmoid(lrf.same_traj_classifier(states, cf_goals)).cpu().numpy()
                    if lrf.disable_ranking:
                        progress = np.ones_like(mask)
                    else:
                        progress = torch.sigmoid(lrf.ranking_network(states, cf_goals)).cpu().numpy()
                    ranking = progress * mask

                else:
                    states_np = traj[0][:-1]
                    if lrf.goal_is_metaworld_style:
                        states_np[:, -3:] = cf_goal
                    states_tensor = torch.Tensor(states_np).float().to(device)

                    mask = torch.sigmoid(lrf.same_traj_classifier(states_tensor)).cpu().numpy()
                    if lrf.disable_ranking:
                        progress = np.ones_like(mask)
                    else:
                        progress = torch.sigmoid(lrf.ranking_network(states_tensor)).cpu().numpy()
                    ranking = progress * mask

                if traj_idx == idx:
                    factual_progress = progress
                    factual_mask = mask
                    factual_ranking = ranking
                    factual_lowdimobs = full_states_np[:, :3]
                else:
                    traj_progresses.append(progress)
                    traj_masks.append(mask)
                    traj_rankings.append(ranking)
                    traj_lowdimobs.append(full_states_np[:, :3])


            plt.clf(); plt.cla()
            for p, handxyz in zip(traj_progresses, traj_lowdimobs):
                plt.scatter(handxyz[:, 0], handxyz[:, 1], c=p, alpha=alpha, cmap=cm.jet, norm=normalize)
            plt.scatter(factual_lowdimobs[:, 0], factual_lowdimobs[:, 1], c=factual_progress, alpha=alpha, cmap=cm.jet, edgecolors='black', norm=normalize)
            plt.colorbar()
            plt.scatter(cf_goal[0, 0], cf_goal[0, 1], marker='x', s=200, color='black')
            set_reward_plot_limits(env_str)
            plt.savefig(f"{progress_dir}/{split_str}_{idx_str}_cftrajs_progress.png")

            plt.clf(); plt.cla()
            for p, handxyz in zip(traj_masks, traj_lowdimobs):
                plt.scatter(handxyz[:, 0], handxyz[:, 1], c=p, alpha=alpha, cmap=cm.jet, norm=normalize)
            plt.scatter(factual_lowdimobs[:, 0], factual_lowdimobs[:, 1], c=factual_mask, alpha=alpha, cmap=cm.jet, edgecolors='black', norm=normalize)
            plt.colorbar()
            plt.scatter(cf_goal[0, 0], cf_goal[0, 1], marker='x', s=200, color='black')
            set_reward_plot_limits(env_str)
            plt.savefig(f"{mask_dir}/{split_str}_{idx_str}_cftrajs_mask.png")

            plt.clf(); plt.cla()
            for p, handxyz in zip(traj_rankings, traj_lowdimobs):
                plt.scatter(handxyz[:, 0], handxyz[:, 1], c=p, alpha=alpha, cmap=cm.jet, norm=normalize)
            plt.scatter(factual_lowdimobs[:, 0], factual_lowdimobs[:, 1], c=factual_ranking, alpha=alpha, cmap=cm.jet, edgecolors='black', norm=normalize)
            plt.colorbar()
            plt.scatter(cf_goal[0, 0], cf_goal[0, 1], marker='x', s=200, color='black')
            set_reward_plot_limits(env_str)
            plt.savefig(f"{ranking_dir}/{split_str}_{idx_str}_cftrajs_ranking.png")


'''
generating plots from replay buffer data
'''
def generate_rb_plots(lrf: LearnedRewardFunction, analysis_dir: Path, num_episodes: int = 20, rb_obs_key: str = "observation", is_input_images: bool = True, kitchen_env_str: int = 0):
    rb = lrf.replay_buffer
    plot_dir = analysis_dir / "rb_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # initialize the replay buffer from disk
    _ = rb._try_fetch_most_recent()

    # replay episodes (output video if images are available)
    for ep_idx in range(num_episodes):
        ep_idx_str = str(ep_idx).zfill(3)

        ep = rb._sample_episode()

        # use cached information
        plt.clf(); plt.cla()
        plt.plot(ep['og_reward'])
        plt.savefig(f"{plot_dir}/{ep_idx_str}_og_reward.png")

        plt.clf(); plt.cla()
        plt.plot(ep['reward'])
        plt.ylim(0, 1)
        plt.savefig(f"{plot_dir}/{ep_idx_str}_learned_reward.png")

        # re-run inference just to make sure the reward function didn't change significantly
        # from when the episode was collected
        if 'metaworld_state_obs' in ep:
            mw_state = ep['metaworld_state_obs'][:-1]
            goals = mw_state[:, -3:]

        if is_input_images:
            img_obs = ep[rb_obs_key][:-1] # CHW
            batch_img_tensor = torch.from_numpy(img_obs).to(device).float()
            batch_img = lrf._preprocess_images(batch_img_tensor)
            batch_goals = torch.from_numpy(goals).to(device).float()
            if lrf.goal_is_image:
                batch_goals = lrf._preprocess_images(batch_goals)

            with torch.no_grad():
                mask = torch.sigmoid(lrf.same_traj_classifier(batch_img, batch_goals)).cpu().numpy()
                if lrf.disable_ranking:
                    progress = np.ones_like(mask)
                else:
                    progress = torch.sigmoid(lrf.ranking_network(batch_img, batch_goals)).cpu().numpy()
                ranking = progress * mask
        else:
            lowdim_obs = ep[rb_obs_key][:-1]
            lowdim_obs_tensor = torch.from_numpy(lowdim_obs).to(device).float()

            with torch.no_grad():
                mask = torch.sigmoid(lrf.same_traj_classifier(lowdim_obs_tensor)).cpu().numpy()
                if lrf.disable_ranking:
                    progress = np.ones_like(mask)
                else:
                    progress = torch.sigmoid(lrf.ranking_network(lowdim_obs_tensor)).cpu().numpy()
                ranking = progress * mask

        plt.clf(); plt.cla()
        plt.plot(ranking)
        plt.ylim(0, 1)
        plt.savefig(f"{plot_dir}/{ep_idx_str}_learned_reward_rerun.png")

        # generate a video
        if is_input_images:
            img_obs = ep[rb_obs_key] # CHW
            img_obs = np.transpose(img_obs, (0, 2, 3, 1))
            img_shape = img_obs.shape[1:]
            render_size = img_shape[1]
            if render_size % 16 != 0:
                render_size = (render_size // 16 + 1) * 16
            video_recorder = TrainVideoRecorder(plot_dir, folder_name="videos", render_size=render_size)
            for frame_idx, img in enumerate(img_obs):
                if frame_idx == 0:
                    video_recorder.init(img)
                else:
                    video_recorder.record(img)
            video_recorder.save(f"{ep_idx_str}.mp4")


if __name__ == "__main__":
    exit()