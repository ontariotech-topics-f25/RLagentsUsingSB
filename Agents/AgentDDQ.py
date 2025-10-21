import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from .NeuralNetworkForQValues import NeuralNetworkForQValues
from ..BaseAgent import BaseAgent
import os, yaml


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        '''
        Store a single transition (s, a, r, s', done)
        '''
        # ✅ Store as uint8 to save memory (normalized later during sampling)
        state = np.array(state, dtype=np.uint8)
        next_state = np.array(next_state, dtype=np.uint8)
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        '''
        Randomly sample a batch of transitions for training
        '''
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # ✅ Normalize to [0,1] and clip rewards for stability
        states = torch.tensor(np.array(states), dtype=torch.float32) / 255.0
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32) / 255.0
        actions = torch.tensor(np.array(actions), dtype=torch.int64)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).clamp(-1, 1)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        return states, actions, rewards, next_states, dones



class AgentDDQ(BaseAgent):
    def __init__(self, **config):
        super().__init__(**config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.learn_count = 0
        self.learn_step_counter = 0

        # start two networks , target should be frozen as we are training only online network
        self.online_network = NeuralNetworkForQValues(
            output=self.number_of_inputs,
            input_channels=self.number_of_stacked_frames,
            input_height=self.input_size_for_resizing,
            input_width=self.input_size_for_resizing,
            freeze=False
        )
        self.target_network = NeuralNetworkForQValues(
            output=self.number_of_inputs,
            input_channels=self.number_of_stacked_frames,
            input_height=self.input_size_for_resizing,
            input_width=self.input_size_for_resizing,
            freeze=True
        )

        # move both to the same device
        self.online_network.to(self.device)
        self.target_network.to(self.device)

        # Replay buffer (pure Python)
        self.replay_buffer = ReplayBuffer(self.replay_buffer_capacity)

        # optimizer and loss
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

    def next_action(self, obs):
        '''
        The obs value coming from env is of type LazyFrames.
        It returns index of a random action if random value is less than epsilon,
        otherwise returns index of considered optimal action using the online network.
        '''
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        obs = obs / 255.0  # ✅ Normalize observation
        q_values = self.online_network(obs)
        return q_values.argmax().item()

    def decay_elps(self):
        '''
        eps_min is minimum exploration the model is required to do.
        We start with most exploration and keep reducing it.
        We can tune these in SuperMarioConfig file.
        '''
        # ✅ Slightly slower decay for better exploration
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def save_model(self):
        '''
        define path in yaml file
        '''
        # Resolve path relative to this file's location (one level above src/)
        base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        save_path = os.path.join(base_dir, self.path_to_save_model)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.online_network.state_dict(), save_path)
        print(f"✅ Model saved to {os.path.abspath(save_path)}")

    def load_model(self, load_path = None):
        '''
        define path in yaml file
        '''
        if load_path is not None:
            model_path = load_path
        else:
            model_path = self.path_to_save_model

        full_path = os.path.join(os.path.dirname(__file__), "..", "..", model_path)
        full_path = os.path.abspath(full_path)

        print(f"✅ Loading model from {full_path}")
        state_dict = torch.load(full_path, map_location=self.device)
        self.online_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)

    def update_target_network(self):
        '''
        We can do a soft update using Polyak averaging as well.
        '''
        # ✅ Ensure sync_network_rate is respected
        if hasattr(self, "sync_network_rate") and self.sync_network_rate > 0:
            if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
                self.target_network.load_state_dict(self.online_network.state_dict())

    def store_in_memory(self, state, action, reward, next_state, done):
        '''
        it is slower with np array so wrap it with tensor
        '''
        self.replay_buffer.add(state, action, reward, next_state, done)

    def choose_action(self, state):
        '''
        Compatibility wrapper for BaseAgent abstract method.
        Simply calls next_action() internally.
        '''
        return self.next_action(state)

    def learn(self):
        '''
        Perform one learning step using Double DQN.
        '''
        # Wait until we have a decent replay buffer before learning added minimum buffer size
        if len(self.replay_buffer) < max(self.batch_size, 10000):
            return

        self.optimizer.zero_grad()

        # Fix update 
        self.update_target_network()

        # sample batch and move everything to device
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = (
            states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            next_states.to(self.device),
            dones.to(self.device)
        )

        # Q-values for current states → Q(s,a)
        q_online = self.online_network(states)
        idx = torch.arange(self.batch_size, device=self.device)
        predicted_q_values = q_online[idx, actions.squeeze()]

        # Q-values for next states → use online to pick action, target to evaluate it (DDQN)
        next_actions = self.online_network(next_states).argmax(dim=1)
        q_target_next = self.target_network(next_states)
        target_q_values = q_target_next[idx, next_actions]
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_elps()

        # Monitor loss occasionally
        if self.learn_step_counter % 100 == 0:
            print(f"Loss: {loss.item():.4f}, Epsilon: {self.epsilon:.3f}")
