import os

import h5py
import numpy as np
from torch.utils.data import Dataset


class H5PyTrajDset(Dataset):
    def __init__(self, save_path: str = None, read_only_if_exists: bool = True, should_print: bool = True):
        if os.path.exists(save_path):
            if read_only_if_exists:
                if should_print:
                    print(f"{save_path} already exists! loading this file instead. you will NOT be able to add to it.")
                self.f = h5py.File(save_path, "r")
            else:
                if should_print:
                    print(f"{save_path} already exists! loading this file instead. you will be able to add to it.")
                self.f = h5py.File(save_path, "r+")
            self.length = len(self.f.keys())
            if should_print:
                print(f"{save_path} already has {self.length} trajectories!")
            self.created = False
        else:
            if should_print:
                print(f"creating new dataset at {save_path}")
            self.f = h5py.File(save_path, "w")
            self.length = 0
            self.created = True

        self.save_path = save_path
        self.read_only_if_exists = read_only_if_exists
        self.state_shape = None
        self.action_shape = None
        self.reward_shape = None
        self.goal_shape = None

    def __del__(self):
        self.f.close()

    def __len__(self):
        return self.length # this is the number of trajectories stored

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError

        return (
            self.f[str(idx)]["s_t"][:],
            self.f[str(idx)]["a_t"][:],
            self.f[str(idx)]["r_t"][:],
            self.f[str(idx)]["g_t"][:],
            self.f[str(idx)]["env_full_s_t"][:],
        )

    def add_traj(self, states, actions, rewards, goals, env_full_states):
        # env_full_states would be the metaworld state space compatible with
        # the expert policies or the full franka kitchen state space

        # supports adding batch size number of trajs at a time!
        # i.e., 0th index should be 1 for single trajectories

        if not self.created and self.read_only_if_exists:
            assert False

        # assumes input is in batches
        # (bs, traj_length, state_size)
        # (bs, traj_length, action_size)
        # (bs, rand_vec_size)
        if self.state_shape is None:
            self.state_shape = states.shape[1:]
            self.action_shape = actions.shape[1:]
            self.reward_shape = rewards.shape[1:]
            self.goal_shape = goals.shape[1:]
            self.env_full_state_shape = env_full_states.shape[1:]

        bs = states.shape[0]
        for b_idx in range(bs):
            add_idx = self.length + b_idx
            grp = self.f.create_group(f'{add_idx}')
            if len(self.state_shape) == 4: # images
                grp.create_dataset("s_t", shape=self.state_shape, dtype=np.uint8)
            else:
                grp.create_dataset("s_t", shape=self.state_shape, dtype=np.float32)
            grp.create_dataset("a_t", shape=self.action_shape, dtype=np.float32)
            grp.create_dataset("r_t", shape=self.reward_shape, dtype=np.float32)
            grp.create_dataset("g_t", shape=self.goal_shape, dtype=np.float32)
            grp.create_dataset("env_full_s_t", shape=self.env_full_state_shape, dtype=np.float32)

            grp["s_t"][:] = states[b_idx]
            grp["a_t"][:] = actions[b_idx]
            grp["r_t"][:] = rewards[b_idx]
            grp["g_t"][:] = goals[b_idx]
            grp["env_full_s_t"][:] = env_full_states[b_idx]

        self.length += bs
        self.f.flush()
