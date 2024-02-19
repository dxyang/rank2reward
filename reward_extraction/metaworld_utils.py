import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VALID_METAWORLD_ENVS = [
    "reach", "push", "door-open",
    "door-close", "assembly", "bin-picking",
    "button-press-topdown", "drawer-open", "hammer"
]

def extract_from_obs(o, env_str):
    if env_str == "reach":
        hand_pos = o[:3]
        goal_pos = o[-3:]
        obj_pos = None
    elif env_str in [
        "push", "door-open", "door-close",
        "assembly", "bin-picking", "button-press-topdown",
        "drawer-open", "hammer"
    ]:
        hand_pos = o[:3]
        goal_pos = o[-3:]
        obj_pos = o[4:7]
    else:
        assert False

    return hand_pos, goal_pos, obj_pos


def process_obs(o, env_str, no_goal: bool = False, initial_state: np.ndarray = None):
    if env_str in ["reach"]:
        if no_goal:
            assert initial_state is None
            ret_array = np.array(o[:3])
        else:
            if initial_state is not None:
                ret_array =  np.concatenate([o[:3], initial_state, o[-3:]])
            else:
                ret_array =  np.concatenate([o[:3], o[-3:]])
    elif env_str in [
        "push", "door-open", "door-close",
        "assembly", "bin-picking", "button-press-topdown",
        "drawer-open", "hammer"
    ]:
        if no_goal:
            assert initial_state is None
            ret_array =  np.array(o[:7])
        else:
            if initial_state is not None:
                ret_array =  np.concatenate([o[:7], initial_state, o[-3:]])
            else:
                ret_array =  np.concatenate([o[:7], o[-3:]])
    else:
        assert False

    return ret_array.astype(np.float32)

def random_reset(env_str, env, goal_pos=None, hand_init=None, obj_pos=None):
    '''
    metaworld environments require us to properly set the goal and hand and obj reset bounds
    consolidating the times this is done to this method to minimize random reset constants floating around
    '''
    env.reset()

    if not ((goal_pos is not None) or (hand_init is not None) or (obj_pos is not None)):
        if env_str == "reach":
            goal_pos = np.random.uniform(low=[-0.3, 0.5, 0.175], high=[0.3, 0.9, 0.175], size=(3,))
            hand_init = np.random.uniform(low=[0., 0.7, 0.175], high=[0., 0.7, 0.175], size=(3,))
            obj_pos = None
        elif env_str == "push":
            goal_pos = np.random.uniform(low=[-0.1, 0.7, 0.015], high=[0.1, 0.9, 0.015], size=(3,))
            hand_init = np.random.uniform(low=[0., 0.4, 0.08], high=[0., 0.4, 0.08], size=(3,))
            obj_pos = np.random.uniform(low=[-0.1, 0.6, 0.02], high=[0.1, 0.7, 0.02], size=(3,))
            # obj_pos = np.random.uniform(low=[0.0, 0.6, 0.02], high=[0.0, 0.6, 0.02], size=(3,))

            # HACK initialize the hand ontop of the object for push
            hack = False
            if hack:
                hand_init = obj_pos
                hand_init[2] += 0.15
        elif env_str in ["door-open", "door-close"]:
            obj_pos = np.random.uniform(low=[0., 0.85, 0.15], high=[0.1, 0.95, 0.15], size=(3,))
            hand_init = None
            goal_pos = None
        elif env_str == "assembly":
            obj_pos = None
            hand_init = None
            goal_pos = np.random.uniform(low=[-0.1, 0.75, 0.1], high=[0.1, 0.85, 0.1], size=(3,))
        elif env_str == "bin-picking":
            obj_pos = np.random.uniform(low=[-0.21, 0.65, 0.02], high=[-0.03, 0.75, 0.02], size=(3,))
            hand_init = None
            goal_pos = None
        elif env_str == "button-press-topdown":
            obj_pos = np.random.uniform(low=[-0.1, 0.8, 0.115], high=[0.1, 0.9, 0.115], size=(3,))
            hand_init = None
            goal_pos = None
        elif env_str == "drawer-open":
            obj_pos = np.random.uniform(low=[-0.1, 0.9, 0.0], high=[0.1, 0.9, 0.0], size=(3,))
            hand_init = None
            goal_pos = None
        elif env_str == "hammer":
            obj_pos = np.random.uniform(low=[-0.1, 0.4, 0.0], high=[0.1, 0.5, 0.0], size=(3,))
            hand_init = None
            goal_pos = None
        else:
            assert False

    o, obj_pos, goal_pos = env.reset_model_ood(obj_pos=obj_pos, goal_pos=goal_pos, hand_pos=hand_init)

    return o, obj_pos, goal_pos

def collect_data(
    env,
    env_str,
    horizon,
    policy,
    num_trajs=None,
    model=None,
    collect_images=False,                 # collect 224x224 images in addition to the low dim state space
    render=False,                         # useful for debugging behavior of the policy
    framestack=False,                     # prepends past states without the goal to the state space; don't mix with adding in the original state or images /shrug
    include_original_state_in_state=False, # adds original goal after current state but before goal; don't mix with framestacking /shrug
):
    """Collect expert rollouts from the environment, store it in some datasets"""
    trajs = []
    success_count = 0
    pbar = tqdm(num_trajs)
    while success_count < num_trajs:
        o, obj_pos, goal_pos = random_reset(env_str, env)

        if include_original_state_in_state:
            initial_state = process_obs(o, env_str, no_goal=True).copy()
        else:
            initial_state = None

        o_tm1_no_goal =  process_obs(o, env_str, no_goal=True)
        o_tm2_no_goal =  process_obs(o, env_str, no_goal=True)

        traj = {'obs': [],'action': [], 'next_obs': [], 'done': [], 'reward': [], 'images': [], 'goals': [], 'tcp': None}
        for t in range(horizon):
            if hasattr(policy,'predict'): #sac
                ac, _ = policy.predict(o, deterministic=True)
            elif hasattr(policy,'get_action'):
                ac = policy.get_action(o)
            else:
                t1s = torch.Tensor(o[None]).to(device)
                ac = policy(t1s).cpu().detach().numpy()[0]

            if collect_images:
                if render:
                    curr_img = np.zeros((224, 224, 3))
                else:
                    curr_img = env.sim.render(224, 224, mode='offscreen', camera_name='topview')
                traj['images'].append(curr_img.copy())

            curr_goal = env._get_obs_dict()['state_desired_goal']
            no, r, done, info = env.step(ac)

            # framestacking
            if framestack:
                assert not collect_images # unsupported yet
                po = process_obs(o, env_str)
                stacked_pobs = np.concatenate([o_tm2_no_goal, o_tm1_no_goal, po]).copy()
                pno = process_obs(no, env_str)
                o_t_no_goal = process_obs(o, env_str, no_goal=True)
                stacked_pnobs = np.concatenate([o_tm1_no_goal, o_t_no_goal, pno]).copy()

                save_obs = stacked_pobs
                save_next_obs = stacked_pnobs

                # bookkeeping
                o_tm2_no_goal = o_tm1_no_goal
                o_tm1_no_goal = o_t_no_goal
            else:
                save_obs = process_obs(o, env_str, initial_state=initial_state).copy()
                save_next_obs = process_obs(no, env_str, initial_state=initial_state).copy()

            traj['obs'].append(save_obs)
            traj['goals'].append(curr_goal.copy())
            traj['action'].append(ac.copy())
            traj['next_obs'].append(save_next_obs)
            traj['done'].append(info['success'])
            # traj['reward'].append(info['in_place_reward'])
            traj['reward'].append(r)

            o = no

            if render:
                env.render()
        if info["success"]:
            success_count += 1
            pbar.update()
        else:
            print(f"unsuccessful episode")
            continue
        traj['obs'] = np.array(traj['obs'])
        traj['action'] = np.array(traj['action'])
        if collect_images:
            traj['images'] = np.transpose(np.array(traj['images']), [0, 3, 1, 2])
        traj['goals'] = np.array(traj['goals'])
        traj['next_obs'] = np.array(traj['next_obs'])
        traj['done'] = np.array(traj['done'])
        traj['reward'] = np.array(traj['reward'])
        traj['tcp'] = env.tcp_center #gripper at end of trajectory TODO not good indicator for all tasks
        trajs.append(traj)

    print(f"{success_count} successfull trajectories out of {num_trajs}")
    return trajs

def set_reward_plot_limits(env_str):
    if env_str == "push":
        plt.xlim((-0.15, 0.15))
        plt.ylim((0.35, 0.95))
    elif env_str == "reach":
        plt.xlim((-0.35, 0.35))
        plt.ylim((0.45, 0.95))
    elif env_str in ["door-open", "door-close"]:
        plt.xlim((-0.4, 0.4))
        plt.ylim((0.4, 1.0))
    elif env_str == "assembly":
        plt.xlim((-0.35, 0.35))
        plt.ylim((0.45, 0.95))
    elif env_str == "bin-picking":
        plt.xlim((-0.5, 0.5))
        plt.ylim((0.45, 0.95))
    elif env_str == "button-press-topdown":
        plt.xlim((-0.5, 0.5))
        plt.ylim((0.35, 1.05))
    elif env_str == "drawer-open":
        plt.xlim((-0.5, 0.5))
        plt.ylim((0.35, 1.05))
    elif env_str == "hammer":
        plt.xlim((-0.5, 0.5))
        plt.ylim((0.35, 0.95))
    else:
        assert False


def plot_and_save_models(
    exp_folder,
    losses,
    losses_std,
    train_iterations,
    val_losses,
    val_iterations,
    ranking_network,
    separate_networks = False,
    losses_same_traj = None,
    losses_std_same_traj = None,
    val_losses_same_traj=None,
    same_classifier = None
):
    losses = np.array(losses)
    losses_std = np.array(losses_std)
    plt.clf(); plt.cla()
    plt.plot(val_iterations, val_losses, label="val", color='orange', alpha=0.5)
    plt.plot(train_iterations, losses, label="train", color='blue')
    plt.fill_between(
        train_iterations,
        losses - losses_std,
        losses + losses_std,
        alpha=0.5,
        color='blue'
    )
    plt.legend()
    plt.savefig(f"{exp_folder}/training_loss.png")
    torch.save(ranking_network.state_dict(), f"{exp_folder}/ranking_policy.pt")

    if separate_networks:
        losses_same_traj = np.array(losses_same_traj)
        losses_std_same_traj = np.array(losses_std_same_traj)
        plt.clf(); plt.cla()
        plt.plot(val_iterations, val_losses_same_traj, label="val", color='orange', alpha=0.5)
        plt.plot(train_iterations, losses_same_traj, label="train", color='blue')
        plt.fill_between(
            train_iterations,
            losses_same_traj - losses_std_same_traj,
            losses_same_traj + losses_std_same_traj,
            alpha=0.5,
            color='blue'
        )
        plt.legend()
        plt.savefig(f"{exp_folder}/training_loss_same_traj.png")
        torch.save(same_classifier.state_dict(), f"{exp_folder}/same_classifier_policy.pt")

    # dump the raw data being plotted
    losses_dict = {
        "loss_ranking": losses,
        "loss_ranking_std": losses_std,
        "train_iterations": train_iterations,
        "val_loss_ranking": val_losses,
        "val_iterations": val_iterations,
        "loss_same_traj": losses_same_traj,
        "loss_same_traj_std": losses_std_same_traj,
        "val_loss_same_traj": val_losses_same_traj,
    }
    pickle.dump(losses_dict, open(f"{exp_folder}/losses.pkl", "wb"))

if __name__ == "__main__":
    pass