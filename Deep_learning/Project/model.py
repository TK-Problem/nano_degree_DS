class Deep_model():
    """

    """
    def __init__(self, data_dir):
        self.load_data(data_dir)


    def load_data(self, data_dir):
        """

        :param data_dir:
        :return:
        """

        #
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        # Define your transforms for the training, validation, and testing sets
        data_train_transforms = transforms.Compose ([transforms.Resize (224),
                                                     transforms.RandomRotation (35),
                                                     transforms.RandomResizedCrop (224),
                                                     transforms.RandomHorizontalFlip (),
                                                     transforms.ToTensor (),
                                                     transforms.Normalize ([0.485, 0.456, 0.406],
                                                                           [0.229, 0.224, 0.225])])
        data_other_transforms = transforms.Compose ([transforms.Resize (224),
                                                     transforms.CenterCrop (224),
                                                     transforms.ToTensor (),
                                                     transforms.Normalize ([0.485, 0.456, 0.406],
                                                                           [0.229, 0.224, 0.225])])

        # Load the datasets with ImageFolder
        image_train_dataset = datasets.ImageFolder (train_dir, transform=data_train_transforms)
        image_valid_dataset = datasets.ImageFolder (valid_dir, transform=data_other_transforms)
        image_test_dataset  = datasets.ImageFolder (test_dir, transform=data_other_transforms)

        # Using the image datasets and the trainforms, define the dataloaders
        self.loader_train = torch.utils.data.DataLoader (image_train_dataset, batch_size=64, shuffle=True)
        self.loader_valid = torch.utils.data.DataLoader (image_valid_dataset, batch_size=32)
        self.loader_test  = torch.utils.data.DataLoader (image_test_dataset, batch_size=32)