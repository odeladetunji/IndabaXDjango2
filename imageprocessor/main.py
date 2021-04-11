""" Main file to run the entire program

"""

import os
from src.data import data_generator
from src.runs import train_validate, test
from src import params as cfg
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)


import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--mode', type=str, default='train',
                    choices={'train', 'test'},
                    help='Baseline Model to train: any of the following {train, test, predict}')

opt = parser.parse_args()


def main():

    global opt
    train_data, validation_data, test_data = data_generator(cfg.train_dir, cfg.test_dir)

    if opt.mode == "train":
        print("******* Training Mode ************ \n")
        train_validate(train_data, validation_data)

    if opt.mode == "test":
        print("******* Testing Mode ************** \n")
        test(cfg.model_path, test_data)


if __name__ == '__main__':
    main()