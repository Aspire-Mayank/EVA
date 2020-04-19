# Self Driving Car
# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

from PIL import Image as PILImage

from io import StringIO
import sys

import gym
from gym import error, spaces, utils
from gym.utils import seeding

#from gym_wikinav.envs.wikinav_env import web_graph


class CarEnv():
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self, state_dim, action_dim, max_action, max_size=1e5, distance=0, angle=0, pos=0):
         self.state_dim = 0
         self.action_dim = 0
         self.max_action = 0
         self.max_size = max_size
         self.distance = distance
         self.angle = angle
         self.pos = pos
    
    def reset(self):
        return obs

    def step(self, action):

        return obs, action, reward, done

     

