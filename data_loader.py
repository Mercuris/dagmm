from torch.utils.data import DataLoader
import numpy as np


class DatasetLoader(object):
    """ Data loader for the credit card fraud dataset """

    def __init__(self, data_path, mode="train", train_ratio=0.5):
        """
        Initializer that splits the data in train/test sets

        :param data_path: (str) Name of the .npz file in which to go fetch the data
        :param mode: (str) 'train' or 'test'
        :param train_ratio: (float) Train/test set split ratio
        """
        self.mode = mode
        data = np.load(data_path)

        # Get the values of the class labels and of the attributes
        labels = data["ccf"][:, -1]
        features = data["ccf"][:, :-1]

        # Frauds
        frauds_data = features[labels == 1]
        frauds_labels = labels[labels == 1]

        # Non frauds
        normal_data = features[labels == 0]
        normal_labels = labels[labels == 0]

        # Shuffle the train data indices randomly
        nb_normal_samples = normal_data.shape[0]
        rand_idx = np.arange(nb_normal_samples)
        np.random.shuffle(rand_idx)
        split_index = int(nb_normal_samples * train_ratio)

        # Build the train set (non frauds only)
        self.train = normal_data[rand_idx[:split_index]]
        self.train_labels = normal_labels[rand_idx[:split_index]]

        # Build the test set (rest of the non frauds + all the frauds)
        self.test = normal_data[rand_idx[split_index:]]  # Non frauds not in train set
        self.test_labels = normal_labels[rand_idx[split_index:]]
        self.test = np.concatenate((self.test, frauds_data), axis=0)  # Add frauds
        self.test_labels = np.concatenate((self.test_labels, frauds_labels), axis=0)

    def __len__(self):
        """ Number of samples in the object dataset """
        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]

    def __getitem__(self, index):
        """ Get method handle """
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
            return np.float32(self.test[index]), np.float32(self.test_labels[index])
        

def get_loader(data_path, batch_size, mode='train', train_ratio=0.5):
    """
    Build and return data loader

    :param data_path: (str) Name of the .npz file in which to go fetch the data
    :param batch_size: (int) Size of the batch
    :param mode: (str) 'train' or 'test'
    :param train_ratio: (float) Train/test set split ratio
    :return: (DataLoader) Torch data loader
    """
    dataset = DatasetLoader(data_path, mode, train_ratio)

    shuffle = False
    if mode == 'train':
        shuffle = True

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
