import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import models
import torchvision.transforms as T
from tqdm import tqdm

from reward_extraction.data import H5PyTrajDset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TCN_IMAGE_SIZE = (299, 299)

class BatchNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        return F.relu(x, inplace=True)

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x, inplace=True)
        return x

class TCNModel(nn.Module):
    def __init__(self):
        super(TCNModel, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        self.inception_upto_mixed_5d = nn.Sequential(*list(self.inception.children())[:10])
        # self.inception_upto_mixed_5d.eval()
        # for param in self.self.inception_upto_mixed_5d.parameters():
        #     param.requires_grad = False

        self.conv2d_1 = BatchNormConv2d(288, 100, kernel_size=3, stride=1)
        self.conv2d_2 = BatchNormConv2d(100, 20, kernel_size=3, stride=1)
        self.spatial_softmax = nn.Softmax2d()
        self.fc = Dense(31 * 31 * 20, 32)

    def forward(self, x):
        x = self.inception_upto_mixed_5d(x)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.spatial_softmax(x)
        x = self.fc(x.view((x.size()[0], -1)))
        return x

def tcn_triplet_sample(episode_length, batch_size, pos_frame_margin = 10, neg_margin_scale = 2):
    anchors = np.random.randint(episode_length, size=(batch_size,))

    # positive frame
    pos_delta = np.random.randint(pos_frame_margin * 2, size=(batch_size,)) - pos_frame_margin
    positives = np.clip(anchors + pos_delta, 0, episode_length)

    # negative frame
    negatives = []
    for idx in anchors:
        neg_frame_margin = neg_margin_scale * pos_frame_margin
        below = np.arange(0, max(0, idx - neg_frame_margin))
        above = np.arange(min(episode_length, idx + neg_frame_margin), episode_length)
        sample_range = np.concatenate([below, above])
        negatives.append(np.random.choice(sample_range))
    negatives = np.array(negatives)

    return anchors, positives, negatives

def distance(a, b):
    diff = torch.abs(a - b)
    return torch.pow(diff, 2).sum(dim=1)


def train_tcn(model, optimizer, expert_data_path, iterations, margin, work_dir):
    expert_data_ptr = H5PyTrajDset(expert_data_path, read_only_if_exists=True)
    expert_data = [d for d in expert_data_ptr]
    num_trajs = len(expert_data)

    bs = 32
    episode_length = 100
    resize = T.Resize(TCN_IMAGE_SIZE)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    losses = []
    steps = []
    embedding_valids, embedding_valids_nomargin = [], []
    plot_frequency = 100

    for i in tqdm(range(iterations)):
        # obtain idxs
        traj_idxs = np.random.randint(num_trajs, size=bs)
        anchor_idxs, pos_idxs, neg_idxs = tcn_triplet_sample(episode_length, bs)

        # extract data
        anchors, poss, negs = [], [], []
        for (traj_idx, anchor_idx, pos_idx, neg_idx) in zip(traj_idxs, anchor_idxs, pos_idxs, neg_idxs):
            states = expert_data[traj_idx][0]
            anchors.append(states[anchor_idx][None])
            poss.append(states[pos_idx][None])
            negs.append(states[neg_idx][None])
        anchors = np.concatenate(anchors)
        poss = np.concatenate(poss)
        negs = np.concatenate(negs)

        # preprocess
        anchor_tensor = normalize(resize(torch.from_numpy(anchors).float().to(device) / 255.0))
        pos_tensor = normalize(resize(torch.from_numpy(poss).float().to(device) / 255.0))
        neg_tensor = normalize(resize(torch.from_numpy(negs).float().to(device) / 255.0))

        # pass through model
        anchor_output = model(anchor_tensor)
        pos_output = model(pos_tensor)
        neg_output = model(neg_tensor)

        # calculate loss
        d_positive = distance(anchor_output, pos_output)
        d_negative = distance(anchor_output, neg_output)
        loss = torch.mean(torch.clamp(d_positive - d_negative + margin, min=0.0))

        embedding_valid = float(torch.sum(d_positive + margin < d_negative)) / d_positive.size()[0]
        embedding_valid_nomargin = float(torch.sum(d_positive < d_negative)) / d_positive.size()[0]

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        steps.append(i)
        losses.append(loss.item())
        embedding_valids.append(embedding_valid)
        embedding_valids_nomargin.append(embedding_valid_nomargin)

        if i % plot_frequency == 0:
            plt.clf(); plt.cla()
            plt.plot(steps, losses)
            plt.savefig(f"{work_dir}/losses.png")

            plt.clf(); plt.cla()
            plt.plot(steps, embedding_valids, label="with margin")
            plt.plot(steps, embedding_valids_nomargin, label="no margin")
            plt.legend()
            plt.savefig(f"{work_dir}/embedding_valid.png")

            torch.save(model.state_dict(), f"{work_dir}/tcn_embedder.pt")


if __name__ == "__main__":
    envs = [
        "assembly", "drawer-open", "hammer", "door-close", "push",
        "reach", "button-press-topdown", "door-open"
    ]

    for env in envs:
        model = TCNModel().to(device)
        lr=1e-4
        optimizer = optim.Adam(list(model.parameters()), lr=lr)

        rewardlearning_vid_repo_root = os.path.expanduser("~/code/rewardlearning-vid")
        task_name = env
        work_dir = f"{rewardlearning_vid_repo_root}/tcn/models/{task_name}"
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        expert_data_path = f"{rewardlearning_vid_repo_root}/ROT/ROT/expert_demos/{task_name}/expert_data.hdf"
        iterations = 5_000
        margin = 2.0
        train_tcn(model, optimizer, expert_data_path, iterations, margin, work_dir)

        torch.save(model.state_dict(), f"{work_dir}/tcn_embedder.pt")