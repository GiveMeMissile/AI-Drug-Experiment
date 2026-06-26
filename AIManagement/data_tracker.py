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
    
    def get_sum(self):
        total = 0
        for episode in self.episodes:
            total += len(episode)
        return total

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

    def get_sample(self, batch_size=hp.BATCH_SIZE, min_episodes=5):
        # Creates a sample of data to be used for training the SimpleNN

        if self.get_sum() < hp.BATCH_SIZE*2:
            return None
        
        if len(self.episodes) < min_episodes:
            return None

        collected_info = []
        sample = []
        for i in range(batch_size):
            # Collect unique parts for the sample for batch_size amount

            loops = 0
            while True:
                episode_idx = random.randint(0, len(self.episodes) - 1)
                episode = self.episodes[episode_idx]

                part_idx = random.randint(0, len(episode) - 1)
                part = episode[part_idx]

                if (episode_idx, part_idx) not in collected_info:
                    collected_info.append((episode_idx, part_idx))
                    sample.append(part)
                    break
                loops += 1
                if loops == 100:
                    return None
        
        return sample
    
    def get_sample_sequence(self, batch_size=hp.BATCH_SIZE, sequence_length=hp.TRAINING_SEQUENCE_LENGTH, min_episodes=16):
        if len(self.episodes) < min_episodes:
            return None
        
        episodes = self.get_valid_episodes(batch_size, sequence_length)
        sequences = []
        for i, episode, in enumerate(episodes):
            if i % 2 == 0:
                sequence = [episode[i:i+sequence_length] for i in range(0, len(episode) - 32, 32)]
            else:
                sequence = [episode[len(episode) - i - 32 : len(episode) - i] for i in range(0, len(episode) - 32, 32)]

            for s in sequence:
                sequences.append(s)
                if len(sequences) == batch_size:
                    return sequences
                
        return sequences

        

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