import math
import time
from typing import Tuple
import random

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms

from drqv2.drqv2 import DrQV2Agent

from reward_extraction.data import H5PyTrajDset
from reward_extraction.models import MLP

from r3m import load_r3m

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InverseDynamicsMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super(InverseDynamicsMLP, self).__init__()
        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dim=512,
            output_dim=output_dim,
            hidden_depth=3,
            output_mod=torch.nn.Tanh()
        )

    def forward(self, s_t: torch.Tensor, s_tp1: torch.Tensor):
        x = torch.cat([s_t, s_tp1], dim=1)
        a_hat = self.mlp(x)
        return a_hat

    def save_model(self, model_path: str):
        print(f"saved InverseDynamicsMLP model to {model_path}")
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path: str):
        print(f"loaded InverseDynamicsMLP model from {model_path}")
        self.load_state_dict(torch.load(model_path))

class InverseDynamicsConvNet(nn.Module):
    def __init__(
        self,
        output_dim: int = 1,
    ):
        super(InverseDynamicsConvNet, self).__init__()

        # nature CNN
        n_input_channels = 6
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.mlp = MLP(
            input_dim=7 * 7 * 64,
            hidden_dim=512,
            output_dim=output_dim,
            hidden_depth=3,
            output_mod=torch.nn.Tanh()
        )

    def forward(self, s_t: torch.Tensor, s_tp1: torch.Tensor):
        x = torch.cat([s_t, s_tp1], dim=1)
        x = self.cnn(x)
        a_hat = self.mlp(x)
        return a_hat

    def save_model(self, model_path: str):
        print(f"saved InverseDynamicsConvNet model to {model_path}")
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path: str):
        print(f"loaded InverseDynamicsConvNet model from {model_path}")
        self.load_state_dict(torch.load(model_path))

class InverseDynamicsR3MNet(nn.Module):
    def __init__(
        self,
        output_dim: int = 1,
        resize_input: bool = True,
        do_multiply_255: bool = True,
        freeze_backbone: bool = True,
    ):
        super(InverseDynamicsR3MNet, self).__init__()
        self.resize_input = resize_input
        self.freeze_r3m = freeze_backbone
        self.do_multiply_255 = do_multiply_255

        self.r3m = load_r3m("resnet18")
        if self.freeze_r3m:
            self.r3m.eval()
            for param in self.r3m.parameters():
                param.requires_grad = False

        self.r3m_embedding_dim = 512
        self.do_multiply_255 = do_multiply_255

        self.mlp = MLP(
            input_dim=2 * self.r3m_embedding_dim,
            hidden_dim=512,
            output_dim=output_dim,
            hidden_depth=3,
            output_mod=torch.nn.Tanh()
        )
        self.resize = transforms.Resize(224)

    def forward(self, s_t: torch.Tensor, s_tp1: torch.Tensor):
        # r3m expects things to be 224 by 224!!!
        if self.resize_input:
            s_t = self.resize(s_t)
            s_tp1 = self.resize(s_tp1)

        # r3m expects things to be [0-255] instead of [0-1]!!!
        if self.do_multiply_255:
            s_t = s_t * 255.0
            s_tp1 = s_tp1 * 255.0

        # some computational savings
        if self.freeze_r3m:
            with torch.no_grad():
                s_t = self.r3m(s_t)
                s_tp1 = self.r3m(s_tp1)
        else:
            s_t = self.r3m(s_t)
            s_tp1 = self.r3m(s_tp1)

        x = torch.cat([s_t, s_tp1], dim=0)

        a_hat = self.mlp(x)
        return a_hat

    def save_model(self, model_path: str):
        print(f"saved InverseDynamicsR3MNet model to {model_path}")
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path: str):
        print(f"loaded InverseDynamicsR3MNet model from {model_path}")
        self.load_state_dict(torch.load(model_path))


class SOIL():
    def __init__(self, state_shape: Tuple[int], action_shape: Tuple[int], rb_loader: torch.utils.data.DataLoader, state_dset_path: str):
        if len(state_shape) == 3:
            self.is_image_obs = True
            self.inverse_model = InverseDynamicsConvNet(output_dim=action_shape[0])
        else:
            self.is_image_obs = False
            self.inverse_model = InverseDynamicsMLP(input_dim=2 * state_shape[0], output_dim=action_shape[0])

        self.inverse_model.to(device)
        self.lr = 1e-4
        self.optimizer = torch.optim.Adam(list(self.inverse_model.parameters()), lr=self.lr)
        self.criterion = torch.nn.MSELoss()
        self.N_INV = 10
        self.inverse_model.eval()

        # from DAPG - https://arxiv.org/pdf/1709.10087.pdf
        self.lambda_0 = 0.1
        self.lambda_1 = 0.95

        # assumes that the batch size used for training the inverse dynamics model is the same
        # as for the RL algorithm, in this case drqv2!
        self.rb_loader = rb_loader
        self._rb_iter = None
        state_dset_file = H5PyTrajDset(state_dset_path)
        self.state_dset = [d for d in state_dset_file] # hack: read into memory is way faster than read into disk
        self.synthetic_a_ts = {}

        self.traj_transition_idxs = []
        for traj_num in range(100):
            for idx in range(100):
                self.traj_transition_idxs.append((traj_num, idx))
        self.traj_transition_idx_iter = iter(self.traj_transition_idxs)

    @property
    def rb_iter(self):
        if self._rb_iter is None:
            self._rb_iter = iter(self.rb_loader)
        return self._rb_iter

    def process_state_only_dataset(self):
        self.inverse_model.eval()

        self.synthetic_a_ts = {}

        for traj_num, traj in enumerate(self.state_dset):
            s_ts = traj[0][:-1]
            s_tp1s = traj[0][1:]

            s_ts_tensor = torch.from_numpy(s_ts).to(device)
            s_tp1s_tensor = torch.from_numpy(s_tp1s).to(device)

            if self.is_image_obs:
                s_ts_tensor = s_ts_tensor.float() / 255.0
                s_tp1s_tensor = s_tp1s_tensor.float() / 255.0

            with torch.no_grad():
                a_hats = self.inverse_model(s_ts_tensor, s_tp1s_tensor).detach()

            self.synthetic_a_ts[traj_num] = a_hats

    def update_inverse_model(self):
        self.inverse_model.train()

        for i in range(1):
            self.optimizer.zero_grad()

            # sample batch from replay buffer
            batch = next(self.rb_iter)
            s_ts, a_ts, _, _, s_tp1s, _, _ = tuple(torch.as_tensor(x, device=device).float() for x in batch)
            if self.is_image_obs:
                s_ts = s_ts / 255.0
                s_tp1s = s_tp1s / 255.0

            # predict actions
            a_t_hats = self.inverse_model(s_ts, s_tp1s)
            loss = self.criterion(a_t_hats, a_ts)
            loss.backward()

            self.optimizer.step()

        self.inverse_model.eval()


    def calculate_policy_loss(self, agent: DrQV2Agent, step):
        loss = 0

        batch_size = 128
        num_batches = 10

        if step % math.floor(10000 / (batch_size * num_batches)) == 0:
            random.shuffle(self.traj_transition_idxs)
            self.traj_transition_idx_iter = iter(self.traj_transition_idxs)

        s_ts = []
        s_tp1s = []
        a_ts = []
        log_probs = []

        # start = time.time()
        calculate_actions_on_the_fly = True
        if calculate_actions_on_the_fly:
            for _ in range(num_batches * batch_size):
                traj_num, transition_idx = next(self.traj_transition_idx_iter)
                states = self.state_dset[traj_num][0][transition_idx: transition_idx + 2]

                s_ts.append(np.expand_dims(states[0], axis=0))
                s_tp1s.append(np.expand_dims(states[1], axis=0))

                if len(s_ts) == batch_size:
                    # generate actions for the states
                    s_ts = torch.from_numpy(np.concatenate(s_ts)).to(device)
                    s_tp1s = torch.from_numpy(np.concatenate(s_tp1s)).to(device)
                    if self.is_image_obs:
                        s_ts = s_ts.float() / 255.0
                        s_tp1s = s_tp1s.float() / 255.0

                    with torch.no_grad():
                        a_t_hats = self.inverse_model(s_ts, s_tp1s).detach()

                    # calculate log prob
                    log_prob = agent.calculate_log_prob(obs=s_ts, step=step, action=a_t_hats)
                    log_probs.append(log_prob)

                    # clear batches
                    s_ts = []
                    s_tp1s = []

                    if len(log_probs) == num_batches:
                        break
        else:
            self.process_state_only_dataset()
            for _ in range(num_batches * batch_size):
                traj_num, transition_idx = next(self.traj_transition_idx_iter)
                s_ts.append(np.expand_dims(self.state_dset[traj_num][0][transition_idx], axis=0))
                a_ts.append(torch.unsqueeze(self.synthetic_a_ts[traj_num][transition_idx], dim=0))

                if len(s_ts) == batch_size:
                    # generate actions for the states
                    s_ts = torch.from_numpy(np.concatenate(s_ts)).to(device)
                    a_ts = torch.cat(a_ts)

                    if self.is_image_obs:
                        s_ts = s_ts.float() / 255.0

                    # calculate log prob
                    log_prob = agent.calculate_log_prob(obs=s_ts, step=step, action=a_ts)
                    log_probs.append(log_prob)

                    # clear batches
                    s_ts = []
                    s_tp1s = []
                    a_ts = []

                    if len(log_probs) == num_batches:
                        break
        # end = time.time()
        # print(f"{end - start} seconds for {len(log_probs)}")

        log_probs = torch.cat(log_probs)
        loss = log_probs.mean()

        modifier = self.lambda_0 * self.lambda_1 ** step
        modified_loss = modifier * loss
        return loss, modified_loss


def plot_and_save_losses():
    pass