#!/usr/bin/env python3
import argparse
from utils import Predictor

if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Inputs for processing images with trained model")

    parser.add_argument("path_to_image", help="mandatory location test image")
    parser.add_argument ("path_to_checkpoint", help="mandatory location model checkpoint")

    # Return top KKK
    parser.add_argument ('--top_k', action="store", type=int, default=5)
    # a mapping of categories to real names
    parser.add_argument ('--category_names', action="store", type=str, default="cat_to_name.json")
    # enable GPU inference
    parser.add_argument ('--gpu', action="store_true", default=False)

    # parse the arguments
    args = parser.parse_args ()
    # store arguments into variables

    path_to_image = args.path_to_image
    path_to_checkpoint = args.path_to_checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu

    # process image
    Predictor(path_to_image, path_to_checkpoint, top_k, category_names, gpu)

    print(path_to_image)
    print(path_to_checkpoint)
    print(top_k)
    print(category_names)
    print(gpu)

