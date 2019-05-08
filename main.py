import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import Model
from data_loader import get_loader
from torch.backends import cudnn
from utils import *


RANDOM_SEED = 13


class Hyperparams:
    # Used to store the hyperparameters of the DAGMM and send them to the model as a single parameter
    def __init__(self, config):
        self.__dict__.update(**config)


def model_setup(config):
    """
    Set up the directories and the data before creating the whole model

    :param config: (Hyperparams) Dictionary of the hyperparameters as a class
    :return: (nn.Module) The model of the DAGMM
    """
    # For fast training
    cudnn.benchmark = True  # Good if input size doesn't change (bad otherwise)

    # Create directories if they don't exist
    make_directory(config.model_save_path)
    make_directory(config.fig_save_path)

    # Create data loader
    data_loader = get_loader(config.data_path, batch_size=config.batch_size, train_ratio=config.train_ratio)

    # Create Model
    return Model(data_loader, vars(config))


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Choose options
    use_all_data = True  # If false, will use an easier sub-set of the data
    train_network = True  # Set to false to test a pretrained network
    pretrained = False  # Set to true if using a pretrained network
    pretrained_name = "/huge_400_dagmm.pth"  # Make sure to change layers_size and encoded_dim below also
    test_network = True  # Set to true to run the tests and save the results
    show_plots = False  # Set to true to see the training and testing plots (will pause the program a few times)
    std_all_features = False  # If true, will standardize all the features (gives worst results)

    # Set hyperparameters and other options
    params = {
        'lr': 1e-4,
        'patience': 10,  # Number of epochs to wait before lowering the lr if there's no improvement in the loss
        'num_epochs': 200,  # Number of training epochs
        'log_step': 10,  # Will show the training metrics every "log_step" epochs
        'model_save_step': 200,  # Save the current state of the model every "model_save_step" epochs (no overwrite)
        'batch_size': 1024,
        'train_ratio': 0.5,  # Training/Testing sets ratio
        'gmm_k': 2,  # Number of component in the GMM (i.e. the number of gaussian distributions)
        'layers_size': (29, 16, 8, 4),  # Define the nb of neurons in each layer and the total nb of layer in the AE
        'encoded_dim': 1,  # Size of the encoded representation of the initial data (last layer of the encoder)
        'lambda_energy': 0.1,  # Value of lambda_1 in the objective function
        'lambda_cov_diag': 0.005,  # Value of lambda_2 in the objective function
        'weight_initialization': False,  # Set to True to initialize the weights with Xavier Uniform instead of default
        'pretrained': pretrained,  # If true, will resume training where it stopped
        'data_path': 'credit_card_fraud.npz',  # Name of the file in which to fetch the data as a numpy array (npz)
        'model_save_path': './dagmm/models',  # Name of the directory under which to save the model
        'fig_save_path': './dagmm/figures',  # Name of the directory under which to save the plots
        'model_load_path': './saved_models' + pretrained_name}  # Name of the directory to load the pretrained model

    # Load dataset and prepare it
    data_path = "./creditcard.csv"  # Name of the file where the data is stored as .csv
    data_npz_save_name = "credit_card_fraud"  # Name of the file under which to save the data as a numpy array (npz)
    data = pd.read_csv(data_path)
    data.drop(["Time"], axis=1, inplace=True)  # In this case, Time attribute is useless

    if use_all_data:  # Original class imbalance
        fraud_percentile = 100 * (len(data["Class"][data["Class"] == 1])) / len(data["Class"])  # 492 frauds

        # Normalize data
        scaler = StandardScaler()  # Standardization with gaussian of mean 0 and variance 1
        if std_all_features:  # Features 1 to 28 are already the result of a PCA
            data.iloc[:, 0:-1] = scaler.fit_transform(data.iloc[:, 0:-1])
        else:  # Standardize only the attribute 29 "Amount"
            data["Amount"] = scaler.fit_transform(data["Amount"].values.reshape(-1, 1))

        # Save data as npz
        np.savez_compressed(data_npz_save_name, ccf=data.values)

    else:  # Simplify dataset for experiments (reduce the class imbalance)
        normal_to_anomaly_ratio = 4
        fraud_percentile = 100 / (normal_to_anomaly_ratio + 1)
        frauds = data[data["Class"] == 1]
        # Do a random sampling of the non fraudulent data to get the new class ratio
        non_frauds = (data[data["Class"] == 0]).sample(n=(normal_to_anomaly_ratio * len(frauds)),
                                                       random_state=RANDOM_SEED)
        easy_data = pd.concat([non_frauds, frauds])

        # Normalize features
        scaler = StandardScaler()  # Standardization with gaussian of mean 0 and variance 1
        if std_all_features:  # Features 1 to 28 are already the result of a PCA
            data.iloc[:, 0:-1] = scaler.fit_transform(data.iloc[:, 0:-1])
        else:  # Standardize only the attribute 29 "Amount"
            data["Amount"] = scaler.fit_transform(data["Amount"].values.reshape(-1, 1))

        # Save data as npz
        np.savez_compressed(data_npz_save_name, ccf=easy_data.values)

    # Build, train and test the model
    model = model_setup(Hyperparams(params))
    if train_network:
        print("\n----------------- Training -----------------")
        train_history = model.train()
        dead_neurons_dict = model.verify_dead_neurons()  # Check if there's dead neurons in the network
        print("Percentage of dead neurons in all layers: {:.1f}%".format(dead_neurons_dict["All Layers"]))
        model.visualize_weights()  # Will show the activation of weights (only the first layer by default)
        train_history.display(show_history=show_plots)  # Save the training plots and show them if desired
    if test_network:
        print("\n------------------- Test -------------------")
        accuracy, precision, recall, f_score = model.test(fraud_percentile=fraud_percentile, show_results=show_plots)
