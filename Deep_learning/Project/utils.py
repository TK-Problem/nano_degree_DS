#!/usr/bin/env python
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json
import numpy as np

class Deep_model():
    """
    Trains and save model to classify flowers
    """
    def __init__(self, data_dir,save_dir,arch,learning_rate,hidden_units,epochs,gpu):
        self.load_data(data_dir)
        print("images loaded")
        self.create_model(arch,hidden_units)
        print("model created")
        self.train_model(arch,learning_rate,epochs,gpu)
        print("model trained")
        save_model(self.deep_model,self.optimizer,arch,save_dir)
        print("model saved")


    def load_data(self, data_dir):
        """
         Creates train, validation and test datasets
        :param
            data_dir: location, where train, validation and test images are located
        """
        # image directories
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        # Define your transforms for the training, validation, and testing sets
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomRotation(35),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        # Load the datasets with ImageFolder
        self.image_datasets = {
            "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
            "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"]),
            "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"]),
        }
        # Using the image datasets and the trainforms, define the dataloaders
        self.dataloaders = {
            "train": torch.utils.data.DataLoader(self.image_datasets["train"], batch_size=64, shuffle=True),
            "valid": torch.utils.data.DataLoader(self.image_datasets["valid"], batch_size=32),
            "test": torch.utils.data.DataLoader(self.image_datasets["test"], batch_size=32)
        }

    def create_model(self,arch,hidden):
        """
        Creates model
        """

        if arch == "vgg16":
            self.deep_model = models.vgg16(pretrained=True)
        elif arch == "resnet18":
            self.deep_model = models.resnet18(pretrained=True)
        elif arch == "densenet161":
            self.deep_model = models.densenet161(pretrained=True)

        # create new classifiers for different models
        classifiers = {
            'vgg16': nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(25088, hidden)),
                ('relu1', nn.ReLU()),
                ('drop1', nn.Dropout(0.5)),
                ('fc2', nn.Linear(hidden, 102)),
                ('output', nn.LogSoftmax(dim=1))
            ])),
            "resnet18": nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(512, 102)),
                ('output', nn.LogSoftmax(dim=1))
            ])),
            "densenet161": nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(2208, hidden)),
                ('relu1', nn.ReLU()),
                ('drop1', nn.Dropout(0.5)),
                ('fc2', nn.Linear(hidden, 102)),
                ('output', nn.LogSoftmax(dim=1))
            ]))

        }

        # Freeze parameters so we don't backprop through them
        for param in self.deep_model.parameters():
            param.requires_grad = False

        # create new classifier for our problem, i.e. rewrite output
        if arch != "resnet18":
            self.deep_model.classifier = classifiers[arch]
        else:
            self.deep_model.fc = classifiers[arch]

    def train_model(self,arch,learning_rate,epochs,gpu):
        """
        Trains model
        """
        # create optimizer
        if arch != "resnet18":
            self.optimizer = optim.Adam(self.deep_model.classifier.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.deep_model.fc.parameters(), lr=learning_rate)

        criterion = nn.NLLLoss()

        # change to cuda if GPU is enabaled
        if gpu == True:
            self.deep_model.to('cuda')

        print_every = 51
        steps = 0

        for e in range(epochs):
            running_loss = 0

            for ii, (inputs, labels) in enumerate(self.dataloaders["train"]):
                steps += 1
                if gpu == True:
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')

                self.optimizer.zero_grad()

                # Forward and backward passes
                outputs = self.deep_model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    # Make sure network is in eval mode for inference
                    self.deep_model.eval()

                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        valid_loss, accuracy = validation(self.deep_model, self.dataloaders["valid"], criterion)

                    print("Epoch: {}/{}... ".format(e + 1, epochs),
                           "Train Loss: {:.4f}".format(running_loss / print_every),
                           "Valid Loss: {:.3f}.. ".format(valid_loss / len(self.dataloaders["valid"])),
                           "Valid Accuracy: {:.3f}".format(accuracy / len(self.dataloaders["valid"])))

                    running_loss = 0

                    # Make sure training is back on
                    self.deep_model.train()

        # Do validation on the test set
        self.deep_model.eval()

        with torch.no_grad():
            test_loss, accuracy = validation(self.deep_model, self.dataloaders["test"], criterion)

        print ("Test Accuracy: {:.3f}".format(accuracy / len(self.dataloaders["test"])))

        self.deep_model.class_to_idx = self.image_datasets["train"].class_to_idx

class Predictor():
    """
    Clasifies
    """
    def __init__(self, path_to_image, path_to_checkpoint, top_k, category_names, gpu):
        self.deep_model = load_model(path_to_checkpoint, gpu)
        print("trained model loaded")
        with open(category_names, 'r') as f:
            self.cat_to_name = json.load(f)
        self.predict_image(path_to_image,top_k,gpu)


    def predict_image(self,path_to_image,top_k,gpu):
        """
        Loads and preprocess image
        """

        # Make prediction
        probs, classes = predict(path_to_image, self.deep_model, topk=top_k, gpu=gpu)

        # get flower names
        top_flowers = [self.cat_to_name[_] for _ in classes]

        # results
        flower_num = path_to_image.split('/')[2]
        print("--Predictions--")
        print("Input flower: {}".format(self.cat_to_name[flower_num]))
        print("Predicted flower name: {}".format(top_flowers[0]))
        print("Probability: {:.3f}".format(probs[0]))
        print("---- Top {} K ----".format(top_k))
        for _ in range(len(top_flowers)):
            print("{}: {:.3f}".format(top_flowers[_],probs[_]))



# Helper functions

def validation(model, validloader, criterion):
    """
    function for the validation pass
    """
    valid_loss = 0
    accuracy = 0
    for images, labels in validloader:
        images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality =(labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

def save_model(model,optimizer,model_arch,checkpoint_path):
    """
    Saves trained model
    """
    checkpoint_path = checkpoint_path + "classifier.pth"
    if model_arch != "resnet18":
        classifier = model.classifier
    else:
        classifier = model.fc

    torch.save({
        'arch': model_arch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer,
        'classifier' : classifier,
        'class_to_idx': model.class_to_idx
    },
    checkpoint_path)

def load_model(checkpoint_path, gpu):
    """
    Loads from checkpoint
    """
    checkpoint = torch.load(checkpoint_path)

    # check architecture
    if checkpoint['arch'] in ["vgg16", "resnet18", "densenet161"]:
        model = getattr(models, checkpoint['arch'])(pretrained=True)
    else:
        print("checkpoint unrecognized")
        return None, None

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # create new classifier for our problem, i.e. rewrite outpu
    if checkpoint['arch'] != "resnet18":
        model.classifier = checkpoint['classifier']
    else:
        model.fc = checkpoint['classifier']

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    if gpu:
        model.to('cuda')

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # resize image with shortest side being 256
    if image.width > 256 or image.height > 256:
        if image.height > image.width:
            factor = 256 / image.height
        else:
            factor = 256 / image.width
        image = image.resize((int(image.width * factor), int(image.height * factor)))

    # crop image
    width, height = image.size  # Get image dimensions
    np_image = np.array(image.crop(((width - 224) / 2, (height - 224) / 2, (width + 224) / 2, (height + 224) / 2)))

    # normalize to 1
    np_image = np_image / 255
    # create means and STD arrays
    means = np.array([[[0.485, 0.456, 0.406] for x in range(224)] for y in range(224)])
    std = np.array([[[0.229, 0.224, 0.225] for x in range(224)] for y in range(224)])
    # substract means, divide by standard deviation and transpose final narray
    return ((np_image - means) / std).transpose((2, 0, 1))

def predict(image_path, model, topk=5, gpu=False):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # open image with PIL
    image = Image.open(image_path)

    # process image
    image = process_image(image)

    # for processing load image to CUDA
    image = torch.from_numpy(image)
    if gpu:
        image = image.to('cuda')

    # add extra dimension of size because models were trained with batches 32
    image.unsqueeze_(0)

    # change data type from double to float
    image = image.float()

    # log predictions
    predictions = model.forward(image)

    # normalize to 1 with exponent
    ps = torch.exp(predictions)

    # get 5 top predictions
    top_probs, top_labels = ps.topk(topk)

    # convert values back to numpy
    top_probs = top_probs.data.cpu().numpy().squeeze().tolist()
    top_labels = top_labels.cpu().numpy()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}

    # get flower names. Convert top_labels to list
    top_labels = [idx_to_class[lab] for lab in top_labels.tolist()]

    return top_probs, top_labels

# python predict.py flowers/test/16/image_06670.jpg classifier.pth --gpu