#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import errno
import os
import sys
import time

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

rng = np.random.RandomState(seed=0)


class QNetwork(nn.Module):
    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.
    def __init__(self, network_type, n_state, n_action):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        super(QNetwork, self).__init__()
        self.network_type = network_type
        self.n_state = n_state
        self.n_action = n_action
        if self.network_type == 'DQN':
            self.fc1 = nn.Linear(self.n_state, 100)
            nn.init.xavier_uniform(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
            self.fc2 = nn.Linear(100, 100)
            nn.init.xavier_uniform(self.fc2.weight.data, gain=nn.init.calculate_gain('relu'))
            self.fc3 = nn.Linear(100, 100)
            nn.init.xavier_uniform(self.fc3.weight.data, gain=nn.init.calculate_gain('relu'))
            self.fc4 = nn.Linear(100, self.n_action)
            nn.init.xavier_uniform(self.fc4.weight.data, gain=nn.init.calculate_gain('linear'))
        elif self.network_type == 'Dueling_DQN':
            self.fc1 = nn.Linear(self.n_state, 100)
            nn.init.xavier_uniform(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
            self.fc2 = nn.Linear(100, 100)
            nn.init.xavier_uniform(self.fc2.weight.data, gain=nn.init.calculate_gain('relu'))
            self.fc3 = nn.Linear(100, 100)
            nn.init.xavier_uniform(self.fc3.weight.data, gain=nn.init.calculate_gain('relu'))
            self.fc4 = nn.Linear(100, 100)
            nn.init.xavier_uniform(self.fc4.weight.data, gain=nn.init.calculate_gain('relu'))
            self.fc5 = nn.Linear(100, 1)
            nn.init.xavier_uniform(self.fc5.weight.data, gain=nn.init.calculate_gain('linear'))
            self.fc6 = nn.Linear(100, self.n_action)
            nn.init.xavier_uniform(self.fc6.weight.data, gain=nn.init.calculate_gain('linear'))
        elif self.network_type == 'Conv_DQN':
            self.conv1 = nn.Conv2d(self.n_state[0], 32, 8, stride=4)
            nn.init.xavier_uniform(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
            nn.init.xavier_uniform(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
            self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
            nn.init.xavier_uniform(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
            conv_size1 = int(np.floor(((np.floor((np.floor((self.n_state[1] - 8) / 4 + 1) - 4) / 2 + 1) - 3) / 1) + 1))
            conv_size2 = int(np.floor(((np.floor((np.floor((self.n_state[2] - 8) / 4 + 1) - 4) / 2 + 1) - 3) / 1) + 1))
            self.fc1 = nn.Linear(64 * conv_size1 * conv_size2, 512)
            nn.init.xavier_uniform(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
            self.fc2 = nn.Linear(512, self.n_action)
            nn.init.xavier_uniform(self.fc2.weight.data, gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        if self.network_type == 'DQN':
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
        elif self.network_type == 'Dueling_DQN':
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            v = F.relu(self.fc3(x))
            a = F.relu(self.fc4(x))
            v = self.fc5(v)
            a = self.fc6(a)
            x = v.expand(v.size(0), self.n_action) + a - a.mean(1).unsqueeze(1).expand(a.size(0), self.n_action)
        elif self.network_type == 'Conv_DQN':
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.fc1(x.view(x.size(0), -1)))
            x = self.fc2(x)
        return x

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
        pass

    def load_model(self, model_file):
        # Helper function to load an existing model.
        pass

    def load_model_weights(self, weight_file):
        # Helper funciton to load model weights.
        pass


class Replay_Memory():
    def __init__(self, memory_size, burn_in, n_state, env_name):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.
        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.n_state = n_state
        self.env_name = env_name
        if self.burn_in > self.memory_size:
            self.burn_in = self.memory_size
        if self.env_name == 'SpaceInvaders-v0':
            self.s = np.zeros((self.memory_size, self.n_state[1], self.n_state[2]), dtype=np.uint8)
            self.s_next = np.zeros(self.memory_size, dtype=np.int32)
        else:
            self.s = np.zeros((self.memory_size, self.n_state))
            self.s_next = np.zeros((self.memory_size, self.n_state))
        self.a = np.zeros(self.memory_size)
        self.r = np.zeros(self.memory_size)
        self.done = np.zeros(self.memory_size)
        self.len = 0
        self.cur = 0
        self.step = 0

    def sample_batch(self, batch_size):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        batch = rng.choice(self.len, size=batch_size, replace=False)
        if self.env_name == 'SpaceInvaders-v0':
            state_i = np.arange(self.n_state[0], 0, -1)
            s_batch = np.stack([self.s[i - state_i + 1, :, :] for i in batch])
            s_next_batch = np.stack([self.s[self.s_next[i - state_i + 1], :, :] for i in batch])
        else:
            s_batch = self.s[batch, :]
            s_next_batch = self.s_next[batch, :]
        a_batch = self.a[batch]
        r_batch = self.r[batch]
        done_batch = self.done[batch]
        return s_batch, a_batch, r_batch, s_next_batch, done_batch

    def append(self, s, a, r, s_next, done):
        # Appends transition to the memory.
        if self.env_name == 'SpaceInvaders-v0':
            if self.len > 0 and self.step == 0:
                padding = np.zeros_like(s, dtype=np.uint8)
                self.step += 1
                for _ in range(self.n_state[0] - 1):
                    self.append(padding, self.a[self.cur - 1], self.r[self.cur - 1], padding, self.done[self.cur - 1])
            self.s[self.cur, :, :] = self.process_s(s)
            self.s_next[self.cur] = (self.cur + 1) % self.memory_size
            self.s[self.s_next[self.cur], :, :] = self.process_s(s_next)
        else:
            self.s[self.cur, :] = s
            self.s_next[self.cur, :] = s_next
        self.a[self.cur] = a
        self.r[self.cur] = r
        self.done[self.cur] = done
        if self.len < self.memory_size:
            self.len += 1
        self.cur = (self.cur + 1) % self.memory_size

    def process_s(self, s):
        x = self.n_state[2]
        y = int(np.round(s.shape[0] * x / s.shape[1]))
        crop = int((y - self.n_state[1]) / 2)
        s = cv2.cvtColor(cv2.resize(s, (x, y))[crop:-crop, :, :], cv2.COLOR_RGB2GRAY)
        return s.reshape((1, self.n_state[1], self.n_state[2]))

    def get_full_s(self, s):
        s = self.process_s(s)
        if self.step > 0:
            state_i = np.arange(min(self.step, self.n_state[0] - 1), 0, -1)
            full_s = np.vstack(
                (self.s[self.cur - state_i, :, :].reshape((len(state_i), self.n_state[1], self.n_state[2])), s))
        else:
            full_s = s
        if self.step < self.n_state[0] - 1:
            padding = np.zeros_like(s, dtype=np.uint8)
            for _ in range(self.n_state[0] - self.step - 1):
                full_s = np.vstack((padding, full_s))
        return full_s.reshape((1, self.n_state[0], self.n_state[1], self.n_state[2]))


class DQN():
    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.
    def __init__(self, network_type, env_name, batch_size, render, video, lr, weight_decay, gamma, epsilon_start,
                 epsilon_end, epsilon_decay_interval, memory_size, burn_in, target_update_frequency):
        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.network_type = network_type
        self.env_name = env_name
        self.batch_size = batch_size
        self.render = render
        self.video = video
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_interval
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.target_update_frequency = target_update_frequency

        self.env = gym.make(self.env_name)
        if self.env_name == 'SpaceInvaders-v0':
            self.n_state = [4, 84, 84]
        else:
            self.n_state = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n
        self.burn_in_env = gym.make(self.env_name)
        self.test_env = gym.make(self.env_name)
        if self.network_type == 'Conv_DQN':
            self.is_cuda = torch.cuda.is_available()
        else:
            self.is_cuda = False
        torch.manual_seed(0)
        if self.is_cuda:
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.deterministic = True
        self.network = QNetwork(network_type=self.network_type, n_state=self.n_state, n_action=self.n_action)
        if self.env_name == 'SpaceInvaders-v0':
            self.target_network = QNetwork(network_type=self.network_type, n_state=self.n_state, n_action=self.n_action)
        if self.is_cuda:
            self.network.cuda()
            if self.env_name == 'SpaceInvaders-v0':
                self.target_network = self.target_network.cuda()
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.memory = Replay_Memory(self.memory_size, self.burn_in, self.n_state, self.env_name)

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        if rng.rand() > self.epsilon:
            a = q_values.max(1)[1].data[0]
        else:
            a = rng.randint(0, self.n_action)
        return a

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        a = q_values.max(1)[1].data[0]
        return a

    def train(self, train_episode, video_interval, valid_interval, valid_episode):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.
        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        self.train_episode = train_episode
        self.video_interval = video_interval
        self.burn_in_memory()
        self.epsilon = self.epsilon_start
        if self.video:
            self.env = gym.wrappers.Monitor(self.env, "./Videos/" + self.network_type + "_" + self.env_name + "/",
                                            force=True, video_callable=self.video_callable)
        train_total_r = []
        valid_total_r_mean = []
        i = 0
        for episode in range(train_episode):
            s = self.env.reset()
            self.memory.step = 0
            done = False
            total_r = 0
            while not done:
                if self.env_name == 'SpaceInvaders-v0' and i % self.target_update_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
                if self.env_name == 'SpaceInvaders-v0':
                    f_s = Variable(torch.FloatTensor(self.memory.get_full_s(s)))
                else:
                    f_s = Variable(torch.FloatTensor(s)).view(-1, self.n_state)
                if self.is_cuda:
                    f_s = f_s.cuda()
                a = self.epsilon_greedy_policy(self.network.forward(f_s))
                s_next, r, done, info = self.env.step(a)
                if self.memory.memory_size > 0:
                    self.memory.append(s, a, r, s_next, done)
                    s_batch, a_batch, r_batch, s_next_batch, done_batch = self.memory.sample_batch(self.batch_size)
                else:
                    s_batch, a_batch, r_batch, s_next_batch, done_batch = s, [a], [r], s_next, [done]
                if self.env_name == 'SpaceInvaders-v0':
                    s_batch = Variable(torch.FloatTensor(s_batch))
                    s_next_batch = Variable(torch.FloatTensor(s_next_batch))
                else:
                    s_batch = Variable(torch.FloatTensor(s_batch)).view(-1, self.n_state)
                    s_next_batch = Variable(torch.FloatTensor(s_next_batch)).view(-1, self.n_state)
                a_batch = Variable(torch.LongTensor(a_batch)).view(-1, 1)
                r_batch = Variable(torch.FloatTensor(r_batch)).view(-1, 1)
                done_batch = Variable(torch.FloatTensor(done_batch)).view(-1, 1)
                if self.is_cuda:
                    s_batch = s_batch.cuda()
                    a_batch = a_batch.cuda()
                    r_batch = r_batch.cuda()
                    s_next_batch = s_next_batch.cuda()
                    done_batch = done_batch.cuda()
                if self.env_name == 'SpaceInvaders-v0':
                    q_s_next = self.target_network(s_next_batch).detach().max(1)[0].view(-1, 1)
                else:
                    q_s_next = self.network(s_next_batch).detach().max(1)[0].view(-1, 1)
                q_s = r_batch + (1 - done_batch) * self.gamma * q_s_next
                q_s_a = self.network.forward(s_batch).gather(1, a_batch)
                self.optimizer.zero_grad()
                loss = self.criterion(q_s_a, q_s)
                loss.backward()
                self.optimizer.step()
                if self.epsilon > self.epsilon_end:
                    self.epsilon -= self.epsilon_decay
                total_r += r
                s = s_next
                i += 1
            print("Training Episode: {:4d} Iteration: {:7d} Total Reward: {:6.1f}".format(episode + 1, i, total_r))
            train_total_r.append([episode + 1, total_r])
            if (episode + 1) % np.floor(train_episode / valid_interval) == 0:
                total_r_mean, _ = self.test(valid_episode)
                valid_total_r_mean.append([episode + 1, total_r_mean])
                if episode + 1 > np.floor(train_episode / valid_interval):
                    try:
                        os.remove(reward_saved_path)
                    except OSError as e:
                        if e.errno != errno.ENOENT:
                            print("Error: {:s} - {:s}".format(e.filename, e.strerror))
                total_rs = list(
                    zip([train_total_r, valid_total_r_mean], ["r-", "b-"], ["Training", "Evaluation (Average 20)"]))
                reward_saved_path = "./" + self.network_type + "_" + self.env_name + "_Reward_" + str(
                    episode + 1) + ".pdf"
                self.plot_total_r(total_rs, train_episode, show=False, save=True, path=reward_saved_path)

    def video_callable(self, episode):
        return episode == 0 or (episode + 1) % np.floor(self.train_episode / self.video_interval) == 0

    def plot_total_r(self, total_rs, episode, show=True, save=False, path="./Total_Rewards.pdf"):
        total_r, styles, labels = zip(*total_rs)
        plt.figure(figsize=(10, 5))
        for i in range(len(total_r)):
            x = [episode for episode, _ in total_r[i]]
            y = [total_r for _, total_r in total_r[i]]
            plt.plot(x, y, styles[i], label=labels[i])
        plt.legend()
        plt.xlim(0, episode)
        if episode <= 10:
            x_ticks = range(0, episode + 1)
        else:
            x_ticks = range(0, episode + 1, int(np.ceil(episode / 10)))
        plt.xticks(x_ticks)
        plt.xlabel("Training Episode")
        plt.ylabel("Total Reward")
        plt.title("Total Reward vs Training Episode")
        if save:
            plt.savefig(path, pad_inches=0)
        if show:
            plt.show()
        else:
            plt.close()

    def test(self, test_episode):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        old_epsilon = self.epsilon
        self.epsilon = self.epsilon_end
        total_r = np.zeros(test_episode)
        i = 0
        for episode in range(test_episode):
            s = self.test_env.reset()
            self.memory.step = 0
            done = False
            while not done:
                if self.render:
                    self.test_env.render()
                if self.env_name == 'SpaceInvaders-v0':
                    f_s = Variable(torch.FloatTensor(self.memory.get_full_s(s)))
                else:
                    f_s = Variable(torch.FloatTensor(s)).view(-1, self.n_state)
                if self.is_cuda:
                    f_s = f_s.cuda()
                a = self.epsilon_greedy_policy(self.network.forward(f_s))
                s_next, r, done, info = self.test_env.step(a)
                total_r[episode] += r
                s = s_next
                i += 1
            print("Test Episode: {:4d} Iteration: {:5d} Total Reward: {:6.1f}".format(episode + 1, i, total_r[episode]))
        self.epsilon = old_epsilon
        total_r_mean = np.mean(total_r)
        total_r_std = np.std(total_r)
        print("Test Average Total Reward: {:.1f}\xB1{:.1f}".format(total_r_mean, total_r_std))
        return total_r_mean, total_r_std

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        self.epsilon = self.epsilon_start
        while self.memory.len < self.memory.burn_in:
            s = self.burn_in_env.reset()
            self.memory.step = 0
            done = False
            while not done:
                if self.env_name == 'SpaceInvaders-v0':
                    f_s = Variable(torch.FloatTensor(self.memory.get_full_s(s)))
                else:
                    f_s = Variable(torch.FloatTensor(s)).view(-1, self.n_state)
                if self.is_cuda:
                    f_s = f_s.cuda()
                a = self.epsilon_greedy_policy(self.network.forward(f_s))
                s_next, r, done, info = self.burn_in_env.step(a)
                if self.memory.len < self.memory.burn_in:
                    self.memory.append(s, a, r, s_next, done)
                    if self.memory.len % 100 == 0:
                        print("burn_in: {:5d}".format(self.memory.len))
                s = s_next


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q-Networks Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    parser.add_argument('--network_type', dest='network_type', type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0)
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--epsilon_start', dest='epsilon_start', type=float, default=0.5)
    parser.add_argument('--epsilon_end', dest='epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon_decay_interval', dest='epsilon_decay_interval', type=float, default=100000)
    parser.add_argument('--train_episode', dest='train_episode', type=int, default=3000)
    parser.add_argument('--video', dest='video', type=int, default=1)
    parser.add_argument('--video_interval', dest='video_interval', type=int, default=3)
    parser.add_argument('--valid_interval', dest='valid_interval', type=int, default=100)
    parser.add_argument('--valid_episode', dest='valid_episode', type=int, default=20)
    parser.add_argument('--test_episode', dest='test_episode', type=int, default=100)
    parser.add_argument('--memory_size', dest='memory_size', type=int, default=50000)
    parser.add_argument('--burn_in', dest='burn_in', type=int, default=10000)
    parser.add_argument('--target_update_frequency', dest='target_update_frequency', type=int, default=10000)
    return parser.parse_args()


def main(args):
    tic = time.time()
    args = parse_arguments()
    agent = DQN(network_type=args.network_type, env_name=args.env, batch_size=args.batch_size, render=args.render,
                video=args.video, gamma=args.gamma, lr=args.lr, weight_decay=args.weight_decay,
                epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
                epsilon_decay_interval=args.epsilon_decay_interval, memory_size=args.memory_size, burn_in=args.burn_in,
                target_update_frequency=args.target_update_frequency)
    agent.train(args.train_episode, args.video_interval, args.valid_interval, args.valid_episode)
    agent.test(args.test_episode)
    print("Finished.")
    toc = time.time()
    print("Elapsed time is {0:.6f} seconds.".format(toc - tic))


if __name__ == '__main__':
    main(sys.argv)
