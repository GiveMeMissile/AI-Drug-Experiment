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


def kill_model(model_number):
    # Used to remove an saved Model file and the JSON metadata attached to it
    idx = -1
    with open(hp.MODEL_INFO) as f:
        model_info = json.load(f)

    if len(model_info["model number"]) == 0:
        return None

    idx = get_idx_from_number(model_info, model_number, "model number")         

    if idx == -1:
        print(f"Model number {model_number} does not exist, please try a model number which actually exists.")
        return None

    model_file = hp.MODEL_DIR + "_" + str(model_number) + ".pth"
    print(model_file)

    if os.path.exists(model_file):
        os.remove(model_file)
    else:
        print(f"Was unable to find model {model_number}\n")

    model_info["model number"].pop(idx)
    model_info["LSTM"].pop(idx)
    model_info["hidden"].pop(idx)
    model_info["layers"].pop(idx)
    model_info["input"].pop(idx)
    model_info["epsilon"].pop(idx)
    with open(hp.MODEL_INFO, 'w') as f:
        json.dump(model_info, f)
        print("The model has been killed")


def get_idx_from_number(data, number, name):
    # Takes in a number from either ai or training metadata and uses it to locate the idx.

    for i, part in enumerate(data[name]):
        if part == number:
            return i
        
    return -1