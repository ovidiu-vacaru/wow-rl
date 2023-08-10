from enviroment.enviroment import WowEnvironment
from api.frame_capture import WindowsFrameCapture, FrameCapture
import time
import sys
import socket
import struct
import random
from neural_network.policies import CNNPolicy
import torch

from torch.distributions.categorical import Categorical


def main():
    policy = CNNPolicy(4)
    env = WowEnvironment(WindowsFrameCapture("World of Warcraft"))

    state = env.starting_position()

    # GOESSSS THROUUUUUGH NEURAL NETWORK AND DECIDES WHAT TO DO AS ACTION

    start_time = time.time()
    # Loop for 20 seconds
    while True:
        action_probs = policy.forward(state)

        # Create a categorical distribution from the action probabilities
        m = Categorical(action_probs)

        # Sample an action from the distribution
        action = m.sample().item()
        print(env.distance)
        state, reward, done = env.step(action)


if __name__ == "__main__":
    main()
