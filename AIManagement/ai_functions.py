from AIManagement import hyperparameters as hp
import json
import os

def check_for_folder():
    # Creates the required model storage directory if it currently does not exists in the system

    if not os.path.exists(hp.MODEL_SAVE_FOLDER):
        os.mkdir(hp.MODEL_SAVE_FOLDER)
    if not os.path.exists(hp.MODEL_INFO):
        open(hp.MODEL_INFO, "x")
        with open(hp.MODEL_INFO, "w") as f:
            json.dump(hp.INFO_FORMAT, f)


def get_lowest(data):
    # Takes in a list of ints and checks for the lowest number which does not exist, (used for model saving).

    if data == []:
        return 1
    data.sort()
    if data[0] > 1:
        return 1
    for i in range(len(data)-1):
        num = data[i]
        upper_num = data[i+1]
        if upper_num - num > 1:
            return num+1
    return max(data) + 1


def train():
    pass