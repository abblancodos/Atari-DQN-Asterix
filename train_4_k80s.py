'''

MIT License

Copyright (c) 2023 Wei Gu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

# Basado en la implementación de https://github.com/iewug para pong, boxing y breakout obtenida de:
# https://github.com/iewug/Atari-DQN


import gymnasium as gym
from model import DQN, DuelDQN
from torch import optim
from utils import Transition, ReplayMemory, VideoRecorder
from wrapper import AtariWrapper
import numpy as np
import random
import torch
import torch.nn as nn
from itertools import count
import os
import matplotlib.pyplot as plt
import math
from collections import deque
import wandb

args = {
    "method": ["dqn", "ddqn"],
    "lr": 2.54e-4,
    "epoch": 6000,
    "batch-size": 32,
    "ddqn_store": True,
    "eval_cycle": 6000,
    "which_gpu": 0,
}

sweep_configuration = {
    'name': 'optimize SVM',
    'method': 'bayes',
    'metric': {'goal': 'minimize', 'name': 'avgloss'},
    'parameters': {
        'batchsize': [32, 64],
        'lr': {'min': 2e-2, 'max': 2e-5},
        'gamma': {'min': 0.9000, 'max': 0.9999},
        'EPS_START': {'min': 1, 'max': 100},
        'EPS_DECAY': {'min': 1000, 'max': 100000},
        'EPS_END': {'min': 0.00005, 'max': 0.05},
        'WARMUP': {'min': 100, 'max': 50000}
    }
}

ddqn = 0
steps_done = 0
eps_threshold = 0
eps_end = 0
eps_start = 0
eps_decay = 0

target_net = None
policy_net = None

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

sweep_id = wandb.sweep(sweep=sweep_configuration, project="AA_Asterix_Optim")


def select_action(state: torch.Tensor) -> torch.Tensor:
    '''
    epsilon greedy
    - epsilon: choose random action
    - 1-epsilon: argmax Q(a,s)

    Input: state shape (1,4,84,84)

    Output: action shape (1,1)
    '''
    global eps_threshold
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
                    math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]]).cuda()


def train():
    loc_batch_size = wandb.config.batchsize
    loc_lr = wandb.config.lr
    loc_gamma = wandb.config.gamma
    loc_eps_start = wandb.config.EPS_START
    loc_eps_decay = wandb.config.EPS_DECAY
    loc_eps_end = wandb.config.EPS_END
    loc_warmup = wandb.config.WARMUP

    global eps_threshold
    global eps_decay
    global eps_start
    global eps_end
    global cuda_device
    global policy_net
    global target_net
    eps_end = loc_eps_end
    eps_start = loc_eps_start
    eps_decay = loc_eps_decay
    eps_threshold = loc_eps_start

    # environment
    env = gym.make("AsterixNoFrameskip-v4")
    env = AtariWrapper(env=env, terminal_on_life_loss=False)

    n_action = env.action_space.n  # pong:6; breakout:4; boxing:18

    # make dir to store result

    log_dir = os.path.join(f"log_asterix", args["method"][ddqn])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, "log.txt")

    # video
    video = VideoRecorder(log_dir)

    # create network and target network
    if ddqn == 0:
        policy_net = DQN(in_channels=4, n_actions=n_action).cuda()
        target_net = DQN(in_channels=4, n_actions=n_action).cuda()
    else:
        policy_net = DuelDQN(in_channels=4, n_actions=n_action).cuda()
        target_net = DuelDQN(in_channels=4, n_actions=n_action).cuda()
    # let target model = model
    try:
        policy_net.load_state_dict(torch.load(os.path.join(log_dir, 'model_continuous.pth')))
    except:
        print("couldn't find initialization file, model started empty")

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # replay memory
    memory = ReplayMemory(50000)

    # optimizer
    optimizer = optim.AdamW(policy_net.parameters(), lr=loc_lr, amsgrad=True)

    # warming up
    print("Warming up...")
    warmupstep = 0
    for epoch in count():
        obs, info = env.reset()  # (84,84)
        obs = torch.from_numpy(obs).cuda()  # (84,84)
        # stack four frames together, hoping to learn temporal info
        obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)  # (1,4,84,84)

        # step loop
        for step in count():
            warmupstep += 1
            # take one step
            action = torch.tensor([[env.action_space.sample()]]).cuda()
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            # convert to tensor
            reward = torch.tensor([reward]).cuda()  # (1)
            done = torch.tensor([done]).cuda()  # (1)
            next_obs = torch.from_numpy(next_obs).cuda()  # (84,84)
            next_obs = torch.stack((next_obs, obs[0][0], obs[0][1], obs[0][2])).unsqueeze(0)  # (1,4,84,84)

            # store the transition in memory
            memory.push(obs, action, next_obs, reward, done)

            # move to next state
            obs = next_obs

            if done:
                break

        if warmupstep > loc_warmup:
            break

    rewardList = []
    lossList = []
    rewarddeq = deque([], maxlen=100)
    lossdeq = deque([], maxlen=100)
    avgrewardlist = []
    avglosslist = []
    # epoch loop
    for epoch in range(args["epoch"]):
        obs, info = env.reset()  # (84,84)
        obs = torch.from_numpy(obs).cuda()  # (84,84)
        # stack four frames together, hoping to learn temporal info
        obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)  # (1,4,84,84)

        total_loss = 0.0
        total_reward = 0

        # step loop
        for step in count():
            # take one step
            action = select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            total_reward += reward
            done = terminated or truncated

            # convert to tensor
            reward = torch.tensor([reward]).cuda()  # (1)
            done = torch.tensor([done]).cuda()  # (1)
            next_obs = torch.from_numpy(next_obs).cuda()  # (84,84)
            next_obs = torch.stack((next_obs, obs[0][0], obs[0][1], obs[0][2])).unsqueeze(0)  # (1,4,84,84)

            # store the transition in memory
            memory.push(obs, action, next_obs, reward, done)

            # move to next state
            obs = next_obs

            # train
            policy_net.train()
            transitions = memory.sample(loc_batch_size)
            batch = Transition(*zip(*transitions))  # batch-array of Transitions -> Transition of batch-arrays.
            state_batch = torch.cat(batch.state)  # (bs,4,84,84)
            next_state_batch = torch.cat(batch.next_state)  # (bs,4,84,84)
            action_batch = torch.cat(batch.action)  # (bs,1)
            reward_batch = torch.cat(batch.reward).unsqueeze(1)  # (bs,1)
            done_batch = torch.cat(batch.done).unsqueeze(1)  # (bs,1)

            # Q(st,a)
            state_qvalues = policy_net(state_batch)  # (bs,n_actions)
            selected_state_qvalue = state_qvalues.gather(1, action_batch)  # (bs,1)

            with torch.no_grad():
                # Q'(st+1,a)
                next_state_target_qvalues = target_net(next_state_batch)  # (bs,n_actions)
                if ddqn:
                    # Q(st+1,a)
                    next_state_qvalues = policy_net(next_state_batch)  # (bs,n_actions)
                    # argmax Q(st+1,a)
                    next_state_selected_action = next_state_qvalues.max(1, keepdim=True)[1]  # (bs,1)
                    # Q'(st+1,argmax_a Q(st+1,a))
                    next_state_selected_qvalue = next_state_target_qvalues.gather(1,
                                                                                  next_state_selected_action)  # (bs,1)
                else:
                    # max_a Q'(st+1,a)
                    next_state_selected_qvalue = next_state_target_qvalues.max(1, keepdim=True)[0]  # (bs,1)

            # td target
            tdtarget = next_state_selected_qvalue * GAMMA * ~done_batch + reward_batch  # (bs,1)

            # optimize
            criterion = nn.SmoothL1Loss()
            loss = criterion(selected_state_qvalue, tdtarget)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # let target_net = policy_net every 1000 steps
            if steps_done % 1000 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                # eval
                if epoch % args["eval_cycle"] == 0:
                    with torch.no_grad():
                        video.reset()
                        evalenv = gym.make("AsterixNoFrameskip-v4")
                        evalenv = AtariWrapper(evalenv, video=video)
                        obs, info = evalenv.reset()
                        obs = torch.from_numpy(obs).cuda()
                        obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)
                        evalreward = 0
                        policy_net.eval()
                        for _ in count():
                            action = policy_net(obs).max(1)[1]
                            next_obs, reward, terminated, truncated, info = evalenv.step(action.item())
                            evalreward += reward
                            next_obs = torch.from_numpy(next_obs).cuda()  # (84,84)
                            next_obs = torch.stack((next_obs, obs[0][0], obs[0][1], obs[0][2])).unsqueeze(
                                0)  # (1,4,84,84)
                            obs = next_obs
                            if terminated or truncated:
                                if info["lives"] == 0:  # real end
                                    break
                                else:
                                    obs, info = evalenv.reset()
                                    obs = torch.from_numpy(obs).cuda()
                                    obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)
                        evalenv.close()
                        video.save(f"{epoch}.mp4")
                        torch.save(policy_net.state_dict(), os.path.join(log_dir, 'model_continuous.pth'))
                        print(f"Eval epoch {epoch}: Reward {evalreward}")
                        # plot loss-epoch and reward-epoch
                        plt.figure(1)
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss")
                        plt.plot(range(len(lossList)), lossList, label="loss")
                        plt.plot(range(len(lossList)), avglosslist, label="avg")
                        plt.legend()
                        plt.savefig(os.path.join(log_dir, "loss.png"))

                        plt.figure(2)
                        plt.xlabel("Epoch")
                        plt.ylabel("Reward")
                        plt.plot(range(len(rewardList)), rewardList, label="reward")
                        plt.plot(range(len(rewardList)), avgrewardlist, label="avg")
                        plt.legend()
                        plt.savefig(os.path.join(log_dir, "reward.png"))
                break

        rewardList.append(total_reward)
        lossList.append(total_loss)
        rewarddeq.append(total_reward)
        lossdeq.append(total_loss)
        avgreward = sum(rewarddeq) / len(rewarddeq)
        avgloss = sum(lossdeq) / len(lossdeq)
        avglosslist.append(avgloss)
        avgrewardlist.append(avgreward)

        output = f"Epoch {epoch}: Loss {total_loss:.2f}, Reward {total_reward}, Avgloss {avgloss:.2f}, Avgreward {avgreward:.2f}, Epsilon {eps_threshold:.2f}, TotalStep {steps_done}"
        print(output)
        with open(log_path, "a") as f:
            f.write(f"{output}\n")
        torch.cuda.empty_cache()

    env.close()
    del target_net
    del policy_net


wandb.agent(sweep_id=sweep_id, count=10, function=train)
