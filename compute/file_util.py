import os
from os.path import sep

import joblib


def get_newest_file(file_absolute_path):
    file_absolute_path = os.path.abspath(file_absolute_path)
    lists = os.listdir(file_absolute_path)
    lists.sort(key=lambda x: os.path.getmtime((file_absolute_path + sep + x)))
    file_new = os.path.join(file_absolute_path, lists[-1])
    return file_new

def save_model(model, absolute_save_path):
    save_dir = os.path.dirname(absolute_save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    joblib.dump(model, absolute_save_path)

def load_model(absolute_save_path):
    return joblib.load(absolute_save_path)

# 'E:\\PycharmProjects\\CalculationService\\save\\density'
if __name__ == '__main__':
    print(get_newest_file("./save/"))