from AIManagement.ai_functions import get_lowest
from AIManagement import hyperparameters as hp
from collections import deque
import random
import pickle
import json
import os

class TrainingData:
    # Class which contains the training data used by the AI during training.

    def __init__(self, max_episodes):
        self.episodes = deque([], maxlen=max_episodes)
        self.new_episode()

    def new_episode(self):
        self.current_episode = []
        self.episodes.append(self.current_episode)
    
    def get_max(self):
        # Gets the max length of the biggest episode

        maximum = 0

        for episode in self.episodes:
            if maximum < len(episode):
                maximum = len(episode)
        
        return maximum

    def save_data(self, old_state, action, new_state, reward, ended):
        # Save the inputted data as a step in the current episode

        self.current_episode.append((old_state, action, new_state, reward, ended))

    def get_valid_episodes(self, num_episodes, size):
        # Gets episodes which contain enough data to be used for training.
        # Also shuffles the data

        episodes = list(self.episodes)
        random.shuffle(episodes)

        valid_episodes = []
        for episode in episodes:
            if len(episode) >= size:
                valid_episodes.append(episode)
            if num_episodes == len(valid_episodes):
                return valid_episodes
        
        return valid_episodes

    def get_sample(self, split_size=hp.BATCH_SIZE/16, batch_size=hp.BATCH_SIZE):
        # Gets a sample from the saved episodes which is used for training.

        if self.get_max() < hp.BATCH_SIZE:
            # print(self.get_max())
            return None

        episodes = self.get_valid_episodes(split_size, batch_size)

        sample = []
        split = batch_size // len(episodes)
        for episode in episodes:
            idx = random.randint(0, len(episode) - split)
            sample += episode[idx : idx + split]
        
        return sample
        

class DataMonitor:
    # Tracks the progress the AI is making in its learning and saves that data to be analyzed later. 

    data = []
    current_sample = {
        "step": [],
        "q values": [],
        "action": [],
        "reward": [],
        "object contacted": [],
        "loss": []
    }

    def __init__(self, model):
        with open(hp.DATA_SAVE_INFO, "r") as f:
            self.info = json.load(f)

        self.load_data(model)

    def add_data(self, steps, q_values, action, reward, object, loss):
        self.current_sample["step"].append(steps)
        self.current_sample["q values"].append(q_values)
        self.current_sample["action"].append(action)
        self.current_sample["reward"].append(reward)
        self.current_sample["object contacted"].append(object)
        self.current_sample["loss"].append(loss)

    def new_episode(self):
        self.data.append(self.current_sample)
        self.current_sample = {
        "step": [],
        "q values": [],
        "action": [],
        "reward": [],
        "object contacted": [],
        "loss": []
    }
        
    def save_data(self, model):
        # Saved the data as a pickle file (:

        if not model in self.info["model"]:
            save_num = get_lowest(self.info["save"])
            self.info["save"].append(save_num)
            self.info["model"].append(model)
            with open(hp.DATA_SAVE_INFO, "w") as f:
                json.dump(self.info, f)
        else:
            save_num = self.info["save"][self.info["model"].index(model)]
        
        filename = hp.DATA_SAVE_DIR + str(save_num) + ".pkl"
        if not os.path.exists(filename):
            open(filename, "x")
        with open(filename, "wb") as f:
            pickle.dump(self.data, f)

    def load_data(self, model):
        # Loads saved data

        if not model in self.info["model"]:
            return None
        idx = self.info["model"].index(model)
        save_number = self.info["save"][idx]
        filename = hp.DATA_SAVE_DIR + str(save_number) + ".pkl"
        with open(filename, "rb") as f:
            self.data = pickle.load(f)