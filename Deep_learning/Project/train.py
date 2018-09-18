#!/usr/bin/env python3
import argparse
from utils import Deep_model


if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Inputs for training image recognition model")

    parser.add_argument("data_directory", help="mandatory location for train, validation and test imagaes")

    # save directory
    parser.add_argument('--save_dir', action="store", type=str, default="")
    # hyperparameters
    parser.add_argument('--arch', action="store", type=str, default="vgg16")
    parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
    parser.add_argument('--hidden_units', action="store", type=int, default=512)
    parser.add_argument('--epochs', action="store", type=int, default=1)
    # enable GPU for training model
    parser.add_argument('--gpu', action="store_true", default=False)

    # parse the arguments
    args = parser.parse_args()
    # store arguments into variables

    # image location
    data_dir = args.data_directory

    # save directory
    save_dir = args.save_dir

    # hyperparameters
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    # enable GPU for training model
    gpu = args.gpu

    # train model
    if arch in ["vgg16","resnet18","densenet161"]:
        Deep_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu)
    else:
        print("Unknown model architecture detected")
        print("Available model architectures: vgg16","resnet18","densenet161")

