# Rank2Reward: Learning Shaped Reward Functions from Passive Video

This codebase is used for experimenting with inferring reward functions from videos. From a passive state trajectory, we learn a reward function based on the notion that ranking video frames encodes task completion information and should provide a dense guidance signal (i.e., reward) to an agent trying to learn the task. We also incorporate data collected by the agent during training as negatives to better shape the reward landscape over the full state space.

It is built around a fork of [DrQ-v2](https://github.com/facebookresearch/drqv2), a model-free off-policy algorithm for image-based continuous control. We utilize environments from [metaworld](https://github.com/dxyang/metaworld/tree/rewardlearning-vid) for benchmarking. We also run simple experiments in a 2D pointmass maze environment using an implementation of [SAC](https://github.com/denisyarats/pytorch_sac).

We also include implementations of baselines
* [GAIL](https://arxiv.org/abs/1606.03476) [Ho & Ermon, 2016]
* [VICE](https://arxiv.org/abs/1805.11686) [Fu & Singh et al, 2018]
* [AIRL](https://arxiv.org/abs/1710.11248) [Fu et al, 2018]
* [SOIL](https://arxiv.org/abs/2004.04650) [Radosavovic et al., 2021]
* [Time Contrastive Networks](https://arxiv.org/abs/1704.06888) [Sermanet et al, 2017]
* [Watch and Match: Supercharging Imitation with Regularized Optimal Transport](https://arxiv.org/abs/2206.15469) [Haldar et al, 2022]

## setup
Note we assume [mujoco 2.1.0](https://github.com/deepmind/mujoco/releases/tag/2.1.0) is installed on the computer. Code tested on a computer running Ubuntu 20.04 with nvidia driver version 525.60.13 and cuda toolkit 12.0 installed. Logging is done through [weights and biases](https://wandb.ai/site).

```
# clone the repo
git@github.com:anonymous/rewardlearning-vid.git
git submodule init
git submodule update

# create a virtual environment (note this repo contains .python-version file specifying the name of a virtual environment to look for)
pyenv virtualenv 3.8.10 rewardlearning-vid-py38
pip install --upgrade pip

# install pytorch (system dependent)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# standard pip dependencies
pip install -r requirements.txt

# install submodule dependencies
cd drqv2; pip install -e .
cd r3m; pip install -e .
cd metaworld; pip install -e .
```

## code structure

* `ROT` - git submodule. fork of [official implementation](https://github.com/siddhanthaldar/ROT) adapted to our codebase
* `drqv2` - git submodule. fork of [official implementation](https://github.com/facebookresearch/drqv2). we use the drqv2 agent from here. other parts of the codebase also pull from the replay buffer
* `metaworld` - git submodule. fork of [metaworld](https://github.com/Farama-Foundation/Metaworld). some improvements to camera rendering and initialization of environments.
* `policy_learning` - main folder. includes code that wraps metaworld environments with learned reward functions as well as the main training script that sets up drqv2 with our framework.
* `pytorch_sac` - git submodule. fork of [this SAC implementation](https://github.com/denisyarats/pytorch_sac) used with 2d pointmass environment defined here as well for quick experimentation in a simple domain
* `r3m` - git submodule. vanilla copy of [official implementation](https://github.com/facebookresearch/r3m). useful for extracting features from images.
* `reward_extraction` - main folder. learned reward function model and training code. some metaworld helper code. some expert data saving code.
* `scripts` - misc scripts. data egress from wandb and plotting results code.
* `tcn` - implementation of time contrastive networks adapted from [here](https://github.com/kekeblom/tcn/tree/master)

## useful code entrypoints proxies

### metaworld experiments
valid env strings: `assembly`,  `drawer-open`, `hammer`, `door-close`, `push`, `reach`, `button-press-topdown`, `door-open`

```
# Rank2Reward
python -m policy_learning.train_v2 --env_str hammer --use_online_lrf --seed 42

# GAIL
python -m policy_learning.train_v2 --env_str hammer --use_online_lrf --train_gail

# AIRL
python -m policy_learning.train_v2 --env_str hammer --use_online_lrf --train_airl

# VICE
python -m policy_learning.train_v2 --env_str hammer --use_online_lrf --train_vice

# SOIL
python -m policy_learning.train_v2 --env_str hammer --train_soil

# TCN
python -m policy_learning.train_v2 --env_str hammer --train_tcn

# ROT
cd ROT/ROT
python train.py task_name=hammer
```

### pointmass maze experiments
```
cd pytorch_sac
python train.py
```
