# We import the app_folder because it is needed by the ConsumerTemplate
from .settings import app_folder
from .models import Info

from sharpie.websocket import ConsumerTemplate
from django.contrib.auth.models import User
from channels.db import database_sync_to_async

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

import cv2
import os
import ezpickle
import gymnasium as gym
import numpy as np
import json
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import TimeLimit

from .agent import DISCRETE_OBSERVATION_SPACE_SIZE, epsilon, get_discrete_state, update_q_table

# Websocker consumer that inherits from the consumer template
class Consumer(ConsumerTemplate):
    # Here you define the variables you need for your experiment
    app_folder = app_folder
    # To separate each rooms, we decided to use dictionaries
    step = {}
    env = {}
    agent = {}
    action = {}
    obs = {}
    
    #new
    total_reward = {}
    reward_log = {}
    preference_model = {}
    use_preference = {}
    preference_trajectories = {}
    waiting_for_preference = {}
    preference_received = {}
    episode_count = {}
    MAX_EPISODES = 10000
    
    @database_sync_to_async
    def update_info(self, action, step):
        new_info = Info(user=self.scope["user"], room=self.room_name, action=action, step=step)
        new_info.save()

    # This function is called during the connection with the browser
    async def process_connection(self):
        # Initialize the number of steps
        self.step[self.room_name] = 0
        #new
        self.total_reward[self.room_name] = 0
        self.reward_log[self.room_name] = []
        #new
        # self.use_preference = self.scope["session"].get("use_preference", True)
        self.use_preference = False #enable preference with True
        self.waiting_for_preference[self.room_name] = False
        self.preference_received[self.room_name] = False
        print("Use preference:", self.use_preference)
        
        #new
        self.episode_count[self.room_name] = 0
        self.writer = SummaryWriter(log_dir=f"runs/{self.room_name}")
        
        #new
        if self.room_name not in self.preference_model:
            self.preference_model[self.room_name] = PreferenceModel()
        
        # Start the environment and the agent
        #old
        self.env[self.room_name] = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=self.scope["session"]['goal_velocity'])#temp
        #new
        # self.env[self.room_name] = gym.make("MountainCar-v0", render_mode=None, goal_velocity=self.scope["session"]['goal_velocity'])#temp
        self.env[self.room_name]._max_episode_steps = 500
        #new PPO
        # if os.path.exists('path/to/ppo_mountaincar.zip'):
        #     self.agent[self.room_name] = PPO.load('path/to/ppo_mountaincar')
        #     self.using_ppo = True
        # else:
        #     self.using_ppo = False
            
        # Load the agent if it does exist
        if os.path.exists(self.static_folder[self.room_name]+'agent.pkl'):
            self.agent[self.room_name] = ezpickle.unpickle_data(self.static_folder[self.room_name]+'agent.pkl')
        else:
            self.agent[self.room_name] = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBSERVATION_SPACE_SIZE + [self.env[self.room_name].action_space.n]))
        
        # Get the first observation, render an image and save it on the server
        observation, info = self.env[self.room_name].reset()
        self.obs[self.room_name] = get_discrete_state(observation)
        cv2.imwrite(self.static_folder[self.room_name]+'step.jpg', self.env[self.room_name].render(), [cv2.IMWRITE_JPEG_QUALITY, 80])#temp

    # This function gets the information sent by the browser and processes it
    async def process_inputs(self, text_data):
        # Decode what has been sent by the user
        text_data_json = json.loads(text_data)
        
        #new
        preference = text_data_json.get("preference", None) if self.use_preference else None
        if preference is not None:
            print(f"Received preference label: {preference}")
            traj = self.preference_trajectories[self.room_name]
            self.preference_model[self.room_name].add_example(traj, preference)
            self.preference_trajectories[self.room_name] = []
            print(f"len: {len(self.preference_model[self.room_name].y)}")
            if len(self.preference_model[self.room_name].y) >= 2:
                predicted_score = self.preference_model[self.room_name].predict(traj)
                with open(self.static_folder[self.room_name] + 'preference_scores.csv', 'a') as f:
                    f.write(f"{predicted_score}\n")
                    # f.write(f"{preference}\n")
            else:
                predicted_score = 0.5
            # print(f"Predicted preference score: {predicted_score}")
            self.use_preference = False
            self.waiting_for_preference[self.room_name] = False
            self.preference_received[self.room_name] = True
            # await self.process_ouputs()
            # await self.process_extras()
            
        else:
            left_action = text_data_json["left"]
            right_action = text_data_json["right"]

            #old
            # # Overwrite the action if needed
            # if left_action:
            #     self.action[self.room_name] = 0
            # elif right_action:
            #     self.action[self.room_name] = 2
            # elif np.random.random() > epsilon or self.scope["session"]['train']==False:
            #     self.action[self.room_name] = np.argmax(self.agent[self.room_name][self.obs[self.room_name]])
            # else:
            #     self.action[self.room_name] = np.random.randint(0, self.env[self.room_name].action_space.n)
            # new
            # old PPO
            if left_action:
                self.action[self.room_name] = 0
            elif right_action:
                self.action[self.room_name] = 2
            elif self.room_name in self.obs and self.obs[self.room_name] is not None:
                if np.random.random() > epsilon or self.scope["session"]['train'] == False:
                    self.action[self.room_name] = np.argmax(self.agent[self.room_name][self.obs[self.room_name]])
                else:
                    self.action[self.room_name] = np.random.randint(0, self.env[self.room_name].action_space.n)
                    print("Random action taken.....")
            else:
                # If obs is missing, fallback to random action
                self.action[self.room_name] = np.random.randint(0, self.env[self.room_name].action_space.n)
            
            #new PPO
            # if left_action:
            #     self.action[self.room_name] = 0
            # elif right_action:
            #     self.action[self.room_name] = 2
            # elif self.using_ppo:
            #     action, _ = self.agent[self.room_name].predict(self.env[self.room_name].render(), deterministic=True)
            #     self.action[self.room_name] = int(action)
            # else:
            #     if self.room_name in self.obs and self.obs[self.room_name] is not None:
            #         if np.random.random() > epsilon or self.scope["session"]['train'] == False:
            #             self.action[self.room_name] = np.argmax(self.agent[self.room_name][self.obs[self.room_name]])
            #         else:
            #             self.action[self.room_name] = np.random.randint(0, self.env[self.room_name].action_space.n)
            #     else:
            #         self.action[self.room_name] = np.random.randint(0, self.env[self.room_name].action_space.n)

    # This function performs a step in the experiment
    async def process_step(self):
        # Perform a step in the environment
        observation, reward, terminated, truncated, info = self.env[self.room_name].step(self.action[self.room_name])
        print(f"Step {self.step[self.room_name]}: Action {self.action[self.room_name]}, Reward {reward}, Terminated {terminated}, Truncated {truncated}")
        # Update the Q-table of the agent
        self.obs[self.room_name] = get_discrete_state(observation)
        
        #new
        self.total_reward[self.room_name] += reward
        #new
        if self.use_preference:
            if self.room_name not in self.preference_trajectories:
                self.preference_trajectories[self.room_name] = []
            self.preference_trajectories[self.room_name].append((self.obs[self.room_name], self.action[self.room_name]))
            
        update_q_table(self.obs[self.room_name], self.action[self.room_name], reward, self.agent[self.room_name], terminated, truncated, observation)
        
        #old
        self.terminated[self.room_name] = (terminated or truncated or observation[0] > 0.5)
    
    # This function generates the rendered image and returns the information sent back to the browser
    async def process_ouputs(self):
        # Render an image and save it on the server
        cv2.imwrite(self.static_folder[self.room_name]+'step.jpg', self.env[self.room_name].render(), [cv2.IMWRITE_JPEG_QUALITY, 80])
        # Store the data into the DB
        await self.update_info(self.action[self.room_name], self.step[self.room_name])#temp
        # Check if the game is over
        if self.terminated[self.room_name]:
            if self.use_preference:
                message = 'halfdone'
                print("halfdone")
            #new q-learning
            elif self.scope["session"]['train']:  # Add this check
                message = 'auto'  # auto-restart flag
                print("done (auto)")
            
            else:
                message = 'done'
                print("done")
        else:
            message = 'not done'
        # Send message to room group
        # The returned value should be a dictionnary with the 
        #   type, 
        #   message, 
        #   step, 
        #   and anything else you would like to send
        return {"type": "websocket.message", 
                "message": message, 
                "step": self.step[self.room_name]}
    
    # This function takes care of anything else we need to do at the end of the request
    async def process_extras(self):
        if self.terminated[self.room_name] and self.use_preference:
            self.waiting_for_preference[self.room_name] = True
        elif self.terminated[self.room_name] and self.use_preference == False:
            #new q-learning ------------------------
            # with open(self.static_folder[self.room_name] + 'reward_log.csv', 'a') as f:
            #     f.write(f"{self.total_reward[self.room_name]}\n")
            # self.writer.add_scalar("Reward/Episode", self.total_reward[self.room_name], self.episode_count[self.room_name]) #tensorboard --logdir=runs
            
            print("Episode is done. Saving agent and plotting rewards.")
            ezpickle.pickle_data(self.agent[self.room_name], self.static_folder[self.room_name]+'agent.pkl', overwrite=True)

            # Update episode count
            self.episode_count[self.room_name] += 1
            print(f"Episode #{self.episode_count[self.room_name]} finished.")

            if self.episode_count[self.room_name] >= self.MAX_EPISODES:
                print("Max episodes reached. Stopping.")
                self.env[self.room_name].close()
                del self.step[self.room_name]
                del self.env[self.room_name]
                del self.obs[self.room_name]
                del self.agent[self.room_name]
                del self.action[self.room_name]
                self.writer.close()
                # return  # Stop here
            else:
                # Restart environment (auto-reset)
                self.total_reward[self.room_name] = 0
                self.step[self.room_name] = 0
                # self.env[self.room_name] = gym.make("MountainCar-v0", render_mode=None, goal_velocity=self.scope["session"]['goal_velocity'])
                # self.env[self.room_name]._max_episode_steps = 1000
                observation, info = self.env[self.room_name].reset()
                self.obs[self.room_name] = get_discrete_state(observation)
                self.terminated[self.room_name] = False
            #old q-learning -----------------------
            #new
            # with open(self.static_folder[self.room_name] + 'reward_log.csv', 'a') as f:
            #     f.write(f"{self.total_reward[self.room_name]}\n")
            
            # print("Episode is done. Saving agent and plotting rewards.")
            # # Delete the variables from memory
            # ezpickle.pickle_data(self.agent[self.room_name], self.static_folder[self.room_name]+'agent.pkl', overwrite=True)
            # # old
            # self.env[self.room_name].close()
            # del self.step[self.room_name]
            # del self.env[self.room_name]
            # del self.obs[self.room_name]
            # del self.agent[self.room_name]
            # del self.action[self.room_name]
            # print("Environment closed and variables deleted.")  
            
            #new
            # self.use_preference = True 

            #new
            # if self.use_preference:
            #     # Ask for label from frontend later — temporary CLI label for now
            #     print("Was this episode good? (1 = good, 0 = bad):")
            #     label = int(input("> "))
            #     traj = self.preference_trajectories[self.room_name]
            #     self.preference_model.add_example(traj, label) 
            #     # Use predicted preference reward for logging or Q-update
            #     predicted_score = self.preference_model.predict(traj)
            #     print(f"Predicted preference score: {predicted_score}")
            #     self.total_reward[self.room_name] = predicted_score * 100  # Scale if needed
            #     del self.preference_trajectories[self.room_name]
            
            #new to restart with no key
            # Auto-reset the environment
            # self.total_reward[self.room_name] = 0
            # self.step[self.room_name] = 0
            # self.env[self.room_name].close()
            # self.env[self.room_name] = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=self.scope["session"]['goal_velocity'])
            # observation, info = self.env[self.room_name].reset()
            # self.obs[self.room_name] = get_discrete_state(observation)
            # cv2.imwrite(self.static_folder[self.room_name]+'step.jpg', self.env[self.room_name].render(), [cv2.IMWRITE_JPEG_QUALITY, 80])
            # No deletion here, just re-init
            # self.terminated[self.room_name] = False
            
        else:
            self.step[self.room_name] += 1
                

class PreferenceModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.X = []
        self.y = []

    def pad_flat_features(self, flat_features, target_length):
        return flat_features + [0] * (target_length - len(flat_features))

    def add_example(self, trajectory, label):
        flat_features = []

        # Unroll states and actions
        for obs, action in trajectory:
            flat_features.extend(list(obs))  # 2 elements
            flat_features.append(action)     # 1 element

        # Calculate how many steps are missing
        max_steps = 200
        current_steps = len(trajectory)
        if current_steps < max_steps:
            last_obs = trajectory[-1][0]
            for _ in range(max_steps - current_steps):
                flat_features.extend(list(last_obs))  # repeat last state
                flat_features.append(0)               # pad action with 0
        print(f"Original steps: {current_steps}, Final padded length: {len(flat_features)}")
        self.X.append(flat_features)
        self.y.append(label)

        if len(self.y) >= 2:
            self.model.fit(self.X, self.y)
            print(f"Model trained with {len(self.y)} examples.")

    def predict(self, trajectory):
        try:
            flat_features = []
            for obs, action in trajectory:
                flat_features.extend(list(obs) + [action])
            max_length = 200 * (len(obs) + 1)
            flat_features = self.pad_flat_features(flat_features, max_length)

            check_is_fitted(self.model)
            return self.model.predict_proba([flat_features])[0][1]
        except NotFittedError:
            print("Model not fitted yet — returning neutral score.")
            return 0.5