import argparse

if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Inputs for training image recognition model")

    parser.add_argument("data_directory", help="mandatory location for train, validation and test imagaes")

    # save directory
    parser.add_argument('--save_dir', action="store", type=str, default="/")
    # hyperparameters
    parser.add_argument('--arch', action="store", type=str, default="vgg16")
    parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
    parser.add_argument('--hidden_units', action="store", type=int, default=512)
    parser.add_argument('--epochs', action="store", type=int, default=2)
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
    run_arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    # enable GPU for training model
    run_mode = args.gpu

