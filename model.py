import time
import matplotlib.pyplot as plt
from history import History
from dagmm import *
from utils import *
from data_loader import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # This is actually used for 3D modeling even though IDE doesn't see it


class Model(object):
    """ Wrapper around the DAGMM class to implement the setup, training and testing logic """

    DEFAULTS = {}

    def __init__(self, data_loader, config):
        """
        Initializer

        :param data_loader: (DatasetLoader) Object implementing the train/test split of the data
        :param config: (dict) Hyperparameters and other options
        """
        self.__dict__.update(Model.DEFAULTS, **config)  # Update class variables with hyperparameters dict
        self.data_loader = data_loader

        # Build DAGMM
        self.dagmm = DaGMM(self.gmm_k, layers_size=self.layers_size, compression_dim=self.encoded_dim)
        if self.weight_initialization:  # Initialize network weights with Xavier Uniform and set biases to 0.01
            self.dagmm.apply(initialize_weights)  # According to the results, it doesn't help

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.patience,
                                                                    threshold=1e-4, verbose=True)

        # Print network
        self.print_network()

        # Send DAGMM on GPU if possible
        if torch.cuda.is_available():
            self.dagmm.cuda()

        # Load pretrained model if specified
        if self.pretrained:
            self.load_pretrained_model()

    def print_network(self):
        """ Print the architecture of the network and its total number of parameters"""
        num_params = 0
        for p in self.dagmm.parameters():
            num_params += p.numel()
        print(self.dagmm)
        print("Number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        """ Load the state of a pretrained model """
        self.dagmm.load_state_dict(torch.load(self.model_load_path))
        print("{} loaded".format(self.model_load_path))

    def train(self):
        """
        Train the DAGMM over all the specified epochs

        :return: (History) Log of the different metrics obtained during training
        """
        self.dagmm.train()
        history = History(self.fig_save_path)

        # If using a pretrained model, resume training from the epoch where it previously stopped
        if self.pretrained:
            starting_epoch = int(self.model_load_path.split('_')[-2])
        else:
            starting_epoch = 0

        # Start training
        time_steps = [time.time()]  # To store and print training times
        for epoch in range(starting_epoch, self.num_epochs):
            objective, sample_energy, recon_error, cov_diag = 0, 0, 0, 0

            # Run training steps for all batches for the current epoch
            for index_of_batch, (input_data, labels) in enumerate(self.data_loader):
                input_data = send_to_gpu(input_data)
                loss_step, energy_step, error_step, cov_step = self.dagmm_step(input_data)
                objective += loss_step
                sample_energy += energy_step
                recon_error += error_step
                cov_diag += cov_step

            # Compute average metrics over all batches
            objective /= index_of_batch + 1
            sample_energy /= index_of_batch + 1
            recon_error /= index_of_batch + 1
            cov_diag /= index_of_batch + 1

            # Scheduler step with the objective function value for the current epoch
            self.scheduler.step(objective.item())

            # Save training metrics for the current epoch
            time_steps.append(time.time())
            history.save(objective.data.item(), sample_energy.data.item(), recon_error.data.item(),
                         cov_diag.data.item(), self.optimizer.param_groups[0]["lr"])

            # Print out log info if it's time
            if (epoch + 1) % self.log_step == 0:
                # Logging
                loss = {'total_loss': objective.data.item(),
                        'sample_energy': sample_energy.item(),
                        'recon_error': recon_error.item(),
                        'cov_diag': cov_diag.item()}

                # Printing log
                log = 'Epoch {}'.format(epoch + 1)
                for tag, value in loss.items():
                    log += " - {}: {:.4f}".format(tag, value)
                log += " - Training time: {:.2f}s".format(time_steps[-1] - time_steps[-self.log_step])
                print(log)

            # Save model parameters if it's time
            if (epoch + 1) % self.model_save_step == 0:
                torch.save(self.dagmm.state_dict(),
                           os.path.join(self.model_save_path, '{}_dagmm.pth'.format(epoch + 1)))

        # Print total training time
        time_steps.append(time.time())
        print("\nTotal training time: {:.2f}s".format(time_steps[-1] - time_steps[0]))

        return history

    def verify_dead_neurons(self):
        """
        Computes percentage of dead neurons in the whole DAGMM and in each layer

        :return: (dict) Percentages of dead neurons for each layer and for all layers
        """
        dead_neurons_stats = {}
        total_neurons = 0
        total_dead_neurons = 0

        for j, layer in enumerate(self.dagmm.layers_with_weights):  # Only need to check Linear layers
            gradients = layer.weight.grad.data.cpu().numpy()  # Array of W x N where W is number of grad coming from the
            # next layer and N is the number of neurons on the current layer
            number_of_neurons = gradients.shape[1]  # For the current layer
            total_neurons += number_of_neurons

            # If a column is full of zeros, it means that all the gradients for one neuron are zero (so dead neuron)
            dead_neurons = np.where(~gradients.any(axis=0))[0]
            total_dead_neurons += len(dead_neurons)
            number_of_dead_neurons = len(dead_neurons)
            dead_neurons_stats["Layer " + str(j + 1)] = number_of_dead_neurons / number_of_neurons

        dead_neurons_stats["All Layers"] = total_dead_neurons / total_neurons

        return dead_neurons_stats

    def visualize_weights(self, all_weights=False):
        """
        Plot and saves the activations of the weights of the first layer in the DAGMM

        :param all_weights: (bool) If true, will also plot the activations of the weights of all the layers
        """
        plot_titles = ("Encoder", "Decoder", "Estimator")
        for j, network in enumerate([self.dagmm.encoder, self.dagmm.decoder, self.dagmm.estimation]):
            weights = []
            for module in network:
                if type(module) == torch.nn.Linear:
                    weights.append(module.weight.cpu())

            # Plot average weights of the first layer of the encoder (indicates importance of each attribute)
            if j == 0:
                tensor = torch.mean(weights[0], 0)
                tensor_as_img = tensor.data.numpy().reshape(1, -1)
                plt.figure()
                plt.imshow(tensor_as_img, interpolation="nearest", cmap="gray")
                plt.xticks(range(0, 29, 2))
                plt.xlabel("Attribute")
                plt.yticks([])
                plt.savefig(self.fig_save_path + "/attributes_weight", bbox_inches="tight")

            # Plot the weights of each layer of the network (encoder, decoder and estimator)
            if all_weights:
                n_cols = 1
                n_rows = len(weights)
                plt.figure(figsize=(n_cols * 3, n_rows * 3))
                plt.suptitle(plot_titles[j])
                for i, tensor in enumerate(weights):
                    tensor_as_img = tensor.data.numpy()
                    plt.subplot(n_rows, n_cols, i + 1)
                    plt.imshow(tensor_as_img, interpolation="nearest", cmap="gray")
                plt.savefig(self.fig_save_path + "/layers_weights", bbox_inches="tight")

    def dagmm_step(self, input_data):
        """
        Training step of the DAGMM for a single batch

        :param input_data: (Tensor) Training data
        :return: (tuple of Tensors) Objective value, average sample energy, reconstruction error and sum of the values
        on the diagonal of the covariance matrix
        """
        # Reset optimizer grads to 0
        self.optimizer.zero_grad()

        # Forward pass
        enc, dec, z, gamma = self.dagmm(input_data)

        # Compute objective function
        objective, sample_energy, recon_error, cov_diag = \
            self.dagmm.objective_function(input_data, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag)

        # Backward pass
        objective.backward()
        self.optimizer.step()

        return objective, sample_energy, recon_error, cov_diag

    def test(self, fraud_percentile=20, show_results=True):
        """
        Predict frauds in the test data and show results

        :param fraud_percentile: (float) The expected percentile of the sample energy of the frauds in the data
        :param show_results: (bool) If true, will plot the graphs
        :return: (tuple of floats) Test accuracy, precision, recall and f-score
        """
        # Compute and store train energy with the final values for phi, mu and cov
        self.dagmm.eval()
        self.data_loader.dataset.mode = "train"
        train_energy = []
        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = send_to_gpu(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            sample_energy, cov_diag = self.dagmm.compute_energy(z, phi=self.dagmm.phi, mu=self.dagmm.mu,
                                                                cov=self.dagmm.cov, size_average=False)
            train_energy.append(sample_energy.data.cpu().numpy())
        train_energy = np.concatenate(train_energy, axis=0)

        # Test the network
        self.dagmm.eval()
        self.data_loader.dataset.mode = "test"
        test_energy = []
        test_labels = []
        test_z = []
        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = send_to_gpu(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            sample_energy, cov_diag = self.dagmm.compute_energy(z, size_average=False)
            test_energy.append(sample_energy.data.cpu().numpy())
            test_z.append(z.data.cpu().numpy())
            test_labels.append(labels.numpy())
        test_energy = np.concatenate(test_energy, axis=0)
        test_z = np.concatenate(test_z, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        # Compute the energy threshold for fraud detection
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)  # Combine train and test energy
        thresh = np.percentile(combined_energy, 100 - fraud_percentile)
        print("Fraud percentile: {:.6f} - Threshold: {:.4f}".format(fraud_percentile, thresh))

        # Predict frauds and get ground truth
        pred = (test_energy >= thresh).astype(int)
        ground_truth = test_labels.astype(int)

        # Compute accuracy, precision, recall and f_score
        accuracy = accuracy_score(ground_truth, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(ground_truth, pred, average='binary')

        # Print metrics
        print("Accuracy: {:0.4f}, Precision: {:0.4f}, Recall: {:0.4f}, F-score: {:0.4f}".format(
            accuracy, precision, recall, f_score))
        print("Confusion matrix (truth on the lines)")
        print(confusion_matrix(ground_truth, pred))
        print("Classification Report")
        print(classification_report(ground_truth, pred, target_names=["Normal", "Fraud"], digits=4))

        # Plot the histogram of the energy of the normal and frauds samples
        # Sets up the axis and gets histogram data
        colors = ["blue", "red"]
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        n, bins, patches = ax1.hist([test_energy[test_labels == 0], test_energy[test_labels == 1]], color=colors)
        ax1.cla()  # clear the axis
        # Plots the histogram data
        width = (bins[1] - bins[0]) * 0.4
        bins_shifted = bins + width
        ax1.grid(axis="y", zorder=1)
        ax1.bar(bins[:-1], n[0], width, align='edge', color=colors[0], zorder=2, label="Normal")
        ax2 = ax1.twinx()
        ax2.bar(bins_shifted[:-1], n[1], width, align='edge', color=colors[1], label="Fraud")
        # Create combined legend
        bars1, lab1 = ax1.get_legend_handles_labels()
        bars2, lab2 = ax2.get_legend_handles_labels()
        ax2.legend(bars1 + bars2, lab1 + lab2, loc='upper center')
        # Finish the plot
        ax1.set_ylabel("Normal count", color=colors[0])
        ax2.set_ylabel("Fraud count", color=colors[1])
        ax1.set_xlabel("Sample energy")
        ax1.tick_params('y', colors=colors[0])
        ax2.tick_params('y', colors=colors[1])
        plt.tight_layout()
        plt.savefig(self.fig_save_path + "/energy_hist")

        if show_results:
            plt.show(block=True)

        # Plot the frauds and non frauds points in the encoded space (encoded x euclidean x cosine)
        if test_z.shape[1] > 3:  # Reduce the encoded dimensions to a single one to be able to visualize in 3D
            pca = PCA(n_components=1)
            x_plot = pca.fit_transform(test_z[:, :-2])
        else:
            x_plot = test_z[:, 0]
        pred_colors = np.where(pred == 1, "r", "b")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_plot, test_z[:, 1], test_z[:, 2], c=test_labels.astype(int), edgecolor=pred_colors, cmap="bwr")
        ax.set_xlabel('Encoded')
        ax.set_ylabel('Euclidean')
        ax.set_zlabel('Cosine')
        plt.savefig(self.fig_save_path + "/3d_representation", bbox_inches="tight")

        if show_results:
            plt.show(block=True)

        return accuracy, precision, recall, f_score
