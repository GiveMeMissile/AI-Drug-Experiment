from AIManagement import hyperparameters as hp
from collections import deque
import random

class TrainingData:
    def __init__(self, max_episodes):
        self.episodes = deque([], maxlen=max_episodes)
        self.current_episode = []
        self.episodes.append(self.current_episode)

    def save_data(self, old_state, action, new_state, reward, ended):
        self.current_episode.append((old_state, action, new_state, reward, ended))

    def get_valid_episodes(self, num_episodes, size):
        episodes = random.shuffle(list(self.episodes))

        valid_episodes = []
        for episode in episodes:
            if len(episode) > size:
                valid_episodes.append(episode)
            if num_episodes == len(valid_episodes):
                return valid_episodes
        
        return valid_episodes

    def get_sample(self, split_size=hp.BATCH_SIZE/16, batch_size=hp.BATCH_SIZE):

        episodes = self.get_valid_episodes(split_size, batch_size)
        sample = []
        split = batch_size // len(episodes)
        for episode in episodes:
            idx = random.randint(0, len(episode) - split - 1)
            sample += episode[idx, idx + split]
        
        return sample
        