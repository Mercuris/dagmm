import matplotlib.pyplot as plt


class History:
    """ Used to log and plot the training metrics of the DAGMM """

    def __init__(self, save_path="./dagmm/figures/train_history"):
        self.history = {
            'loss': [],
            'sample_energy': [],
            'recon_error': [],
            'cov_diag': [],
            'lr': []
        }
        self.save_path = save_path

    def save(self, loss, sample_energy, recon_error, cov_diag, lr):
        self.history['loss'].append(loss)
        self.history['sample_energy'].append(sample_energy)
        self.history['recon_error'].append(recon_error)
        self.history['cov_diag'].append(cov_diag)
        self.history['lr'].append(lr)

    def display_loss(self):
        epoch = len(self.history['loss'])
        epochs = [x for x in range(1, epoch + 1)]
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(epochs, self.history['loss'])
        plt.show()

    def display_sample_energy(self):
        epoch = len(self.history['sample_energy'])
        epochs = [x for x in range(1, epoch + 1)]
        plt.title('Training Sample Energy')
        plt.xlabel('Epochs')
        plt.ylabel('Sample Energy')
        plt.plot(epochs, self.history['sample_energy'])
        plt.show()

    def display_lr(self):
        epoch = len(self.history['loss'])
        epochs = [x for x in range(1, epoch + 1)]
        plt.title('Learning rate')
        plt.xlabel('Epochs')
        plt.ylabel('Lr')
        plt.plot(epochs, self.history['lr'], label='Lr')
        plt.show()

    def display(self, show_history=False):
        """ Plot and save the loss, sample energy and learning rate """
        epoch = len(self.history['loss'])
        epochs = [x for x in range(1, epoch + 1)]

        fig, axes = plt.subplots(3, 1, sharex=True)
        plt.tight_layout()

        axes[0].set_ylabel('Loss')
        axes[0].plot(epochs, self.history['loss'])

        axes[1].set_ylabel('Sample Energy')
        axes[1].plot(epochs, self.history['sample_energy'])

        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('Lr')
        axes[2].set_yscale('log')
        axes[2].plot(epochs, self.history['lr'])

        plt.savefig(self.save_path + "/train_history", bbox_inches="tight")
        if show_history:
            plt.show(block=True)
