import torch.nn as nn
from utils import *


class DaGMM(nn.Module):
    """ Deep Autoencoding Gaussian Mixture Model """

    def __init__(self, n_gmm=2, layers_size=(29, 16, 8, 4), compression_dim=1):
        """
        Initializer

        :param n_gmm: (int) Number of component of the GMM
        :param layers_size: (tuple of int) Configuration of the layers of the AE
        :param compression_dim: (int) Number of encoded dimensions
        """
        super(DaGMM, self).__init__()

        # Dimension of the low-dim representation to feed to the estimation network
        latent_dim = 2 + compression_dim  # 2 is the number of attributes of the reconstruction error

        # Build the network
        self.layers_with_weights = []  # Store all the linear layers (used to find the dead neurons)

        # Encoder
        layers = []  # Used to temporarily store the layers of each part of the DAGMM
        for i in range(len(layers_size) - 1):
            layers += [nn.Linear(layers_size[i], layers_size[i + 1])]
            self.layers_with_weights.append(layers[-1])
            layers += [nn.Tanh()]
        layers += [nn.Linear(layers_size[-1], compression_dim)]
        self.layers_with_weights.append(layers[-1])
        self.encoder = nn.Sequential(*layers)

        # Decoder
        layers = []
        layers += [nn.Linear(compression_dim, layers_size[-1])]
        self.layers_with_weights.append(layers[-1])
        for i in range((len(layers_size) - 1), 0, -1):
            layers += [nn.Tanh()]
            layers += [nn.Linear(layers_size[i], layers_size[i - 1])]
            self.layers_with_weights.append(layers[-1])
        self.decoder = nn.Sequential(*layers)

        # Estimation Neural Network
        layers = []
        layers += [nn.Linear(latent_dim, 10)]
        self.layers_with_weights.append(layers[-1])
        layers += [nn.Tanh()]
        layers += [nn.Dropout(p=0.5)]  # Optionnal
        layers += [nn.Linear(10, n_gmm)]
        self.layers_with_weights.append(layers[-1])
        layers += [nn.Softmax(dim=1)]
        self.estimation = nn.Sequential(*layers)

        # Initialize GMM parameters to 0
        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm, latent_dim, latent_dim))

    def forward(self, x):
        """
        Forward pass of the network

        :param x: (Tensor) Input data
        :return: (tuple of Tensors) Encoded data, decoded data, low-dim data and gamma (exit of the softmax)
        """
        enc = self.encoder(x)  # z_c : Encoded representation learned by the deep AE
        dec = self.decoder(enc)  # x' : Reconstructed counterpart of x

        rec_euclidean = relative_euclidean_distance(x, dec)  # relative euclidean distance metric between x and x'
        rec_cosine = nn.functional.cosine_similarity(x, dec, dim=1)  # cosine distance metric between x and x'

        # z is the combination of z_c and the 2 distance metrics above (to be fed to the estimation network)
        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)

        gamma = self.estimation(z)  # Soft mixture-component membership prediction

        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        """
        Estimate GMM parameters and update the corresponding internal variables

        :param z: (Tensor) Low-dimensional representation of input given by the decoder and the distance metrics
        :param gamma: (Tensor) K-dimensional vector membership prediction
        :return: (tuple of Tensors) phi, mu and covariance matrix parameters of the GMM
        """
        # Compute phi
        n = gamma.size(0)  # Number of samples
        sum_gamma = torch.sum(gamma, dim=0)  # Vector of size 1 x K where K is the number of components of the GMM
        phi = (sum_gamma / n)  # One phi value per component of the GMM
        self.phi = phi.data

        # Compute mu
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)  # Size K x D
        self.mu = mu.data

        # Compute covariance matrix
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))  # z - mu, size of N x K x D where D is the number of attributes of z
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)  # Size of N x K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data  # The size of the cov matrix is K x D x D

        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        """
        Compute the sample energy and the sum of the entries on the diagonal of the covariance matrix

        :param z: (Tensor) Low-dimensional representation of input given by the encoder and the distance metrics
        :param phi: (Tensor) Estimation of the phi parameter for the GMM
        :param mu: (Tensor) Estimation of the mu parameter for the GMM
        :param cov: (Tensor) Estimation of the covariance matrix for the GMM
        :param size_average: (bool) If true, will compute the average of the energy of the samples
        :return: Sample energy (scalar) and the sum of the entries of the covariance matrix (scalar)
        """
        # Send the GMM parameters to GPU the first time this method is called
        if phi is None:
            phi = send_to_gpu(self.phi)
        if mu is None:
            mu = send_to_gpu(self.mu)
        if cov is None:
            cov = send_to_gpu(self.cov)

        # Store useful variables
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))  # z - mu
        eps = 1e-12  # Small constant for numerical stability

        # Initialize the determinant and the diagonal of the covariance matrix
        det_cov = []
        cov_diag = 0

        # Compute covariance matrix for all K samples
        cov_inverse = torch.inverse(cov)  # shape K x D x D
        for i in range(cov.shape[0]):
            cov_diag += torch.sum(torch.diag(cov[i, :, :]))  # Sum of diagonal
            det_cov.append(torch.det(cov[i, :, :]).unsqueeze(0))  # Det falls to 0 when no singular values for cov
        det_cov = torch.cat(det_cov).cuda()  # shape K

        # Compute numerator of the sample energy sum
        exp_term = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        max_val = torch.max(exp_term.clamp(min=0), dim=1, keepdim=True)[0]  # for stability
        exp_term = torch.exp(exp_term - max_val)

        # Compute sample energy
        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if size_average:  # Should be True during training and False during testing
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def objective_function(self, x, x_recon, z, gamma, lambda_energy, lambda_cov_diag):
        """
        Compute the objective function of the network

        :param x: (Tensor) Input data of the compression network
        :param x_recon: (Tensor) Reconstructed data from the decoder
        :param z: (Tensor) Low-dimensional representation of input given by the encoder and the distance metrics
        :param gamma: (Tensor) Soft mixture-component membership predictions
        :param lambda_energy: (float) Value of the lambda_1 factor in the objective function
        :param lambda_cov_diag: (float) Value of the lambda_2 factor in the objective function
        :return: (tuple of Tensors) Objective function value, mean sample energy, reconstruction error and the
        sum of the values on the diagonal of the covariance matrix
        """
        # Compute reconstruction error, GMM params, sample energy and the sum of diagonal of the covariance matrix
        recon_error = torch.mean((x - x_recon) ** 2)  # L2 norm for reconstruction error
        phi, mu, cov = self.compute_gmm_params(z, gamma)  # Estimate GMM parameters (and update internal states)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)

        # Compute the value of objective function
        obj = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return obj, sample_energy, recon_error, cov_diag
