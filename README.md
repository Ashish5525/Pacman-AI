
# Pacman-AI

This repository contains the implementation of a Deep Q-Network (DQN) to play the Ms. Pacman game using PyTorch and Gymnasium (formerly OpenAI Gym). The model uses a Convolutional Neural Network (CNN) to approximate the Q-values for each action and train the agent to play the game efficiently.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Network Architecture](#network-architecture)
- [Visualizing Results](#visualizing-results)
- [References](#references)

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Install the necessary packages**:
    ```sh
    pip install gymnasium
    pip install "gymnasium[atari, accept-rom-license]"
    apt-get install -y swig
    pip install "gymnasium[box2d]"
    pip install torch torchvision
    ```

## Usage

1. **Import the libraries**:
    ```python
    import os
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from collections import deque
    from torch.utils.data import DataLoader, TensorDataset
    ```

2. **Define the Network Architecture**:
    ```python
    class Network(nn.Module):
        def __init__(self, action_size, seed=42):
            super(Network, self).__init__()
            self.seed = torch.manual_seed(seed)
            self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
            self.bn4 = nn.BatchNorm2d(128)
            self.fc1 = nn.Linear(10 * 10 * 128, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, action_size)

        def forward(self, state):
            x = F.relu(self.bn1(self.conv1(state)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    ```

4. **Visualizing Results**:
    - Save and display the trained model playing the game.

    ```python
    import glob
    import io
    import base64
    import imageio
    from IPython.display import HTML, display
    from gym.wrappers.monitoring.video_recorder import VideoRecorder

    def show_video_of_model(agent, env_name):
        env = gym.make(env_name, render_mode='rgb_array')
        state, _ = env.reset()
        done = False
        frames = []
        while not done:
            frame = env.render()
            frames.append(frame)
            action = agent.act(state)
            state, reward, done, _, _ = env.step(action)
        env.close()
        imageio.mimsave('video.mp4', frames, fps=30)

    show_video_of_model(agent, 'MsPacmanDeterministic-v0')

    def show_video():
        mp4list = glob.glob('*.mp4')
        if len(mp4list) > 0:
            mp4 = mp4list[0]
            video = io.open(mp4, 'r+b').read()
            encoded = base64.b64encode(video)
            display(HTML(data=''''''.format(encoded.decode('ascii'))))
        else:
            print("Could not find video")

    show_video()
    ```

## Network Architecture

The network architecture consists of four convolutional layers followed by three fully connected layers:
- Conv1: 32 filters, 8x8 kernel, stride 4
- Conv2: 64 filters, 4x4 kernel, stride 2
- Conv3: 64 filters, 3x3 kernel, stride 1
- Conv4: 128 filters, 3x3 kernel, stride 1
- FC1: 512 units
- FC2: 256 units
- FC3: `action_size` units (output layer)

## Training the Agent

The agent is trained using the Deep Q-Learning algorithm with experience replay. The training process involves:
- Interacting with the environment to collect experiences.
- Storing experiences in a replay buffer.
- Sampling random minibatches from the buffer to train the network.
- Updating the target network periodically.

## Visualizing Results

After training, the performance of the agent can be visualized by saving and displaying a video of the agent playing the game.


https://github.com/Ashish5525/Pacman-AI/assets/70592535/b3f9418d-7a8d-498a-a509-16e3e6292929



https://github.com/Ashish5525/Pacman-AI/assets/70592535/66016b61-3616-4ce7-ad38-6cdc84200a3d

