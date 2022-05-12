import os
import time
from types import ModuleType
from typing import Callable, Union

import anndata
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
from anndata import AnnData
from numpy.random import normal
from torch import tensor

from .nn_losses import weighted_mse

torch.manual_seed(0)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class DeepInterpolation:
    def __init__(
        self,
        model: ModuleType,
        data_sampler: object,
        sirens: bool = False,
        enforce_positivity: bool = False,
        loss_function: Union[Callable, None] = weighted_mse(),
        smoothing_factor: Union[float, None] = True,
        stability_factor: Union[float, None] = True,
        load_model_from_buffer: bool = False,
        buffer_path: str = "model_buffer/",
        hidden_features: int = 256,
        hidden_layers: int = 3,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        **kwargs,
    ):
        """The DeepInterpolation class. Code originally written for dynode (by Arman Rahimzamani), adapted by Xiaojie
        Qiu for building 3d continous models from discrete 3d points cloud.

        Args:
            model: Imported Python module
               Contains A, B, h and MainFlow blocks. Default is interpolation_nn.py
            data_sampler: Data Sampler Object.
            sirens: Whether to use the sinusoidal representation network (SIRENs).
            enforce_positivity: Enforce the positivity of the dynamics when training the networks
            loss_function: The PyTorch-compatible module calculating the differentiable loss function for training by
                the data upon calling.
            smoothing_factor: The coefficient of the Lipschitz smoothness regularization term (added to the loss
                function).
            stability_factor: The coefficient of the Lyapunov stability regularization term (added to the loss function)
            load_model_from_buffer: Whether load the A, B and h module from the saved buffer, or create and initialize
                fresh copies to train.
            buffer_path: The directory address keeping all the saved/to-be-saved torch variables and NN modules.
            hidden_features: The dimension of the hidden layers.
            hidden_layers: The number of total hidden layers.
            first_omega_0: The omega for the first layer used in the SIRENs.
            hidden_omega_0: The omega for the hidden layer used in the SIRENs.
            **kwargs: Additional keyword parameters. Currently not used.
        """

        try:
            os.makedirs(buffer_path)
        except:
            pass

        # TODO: os.errno removed in recent versions. Debug the exception below.

        # try:
        #     os.makedirs(buffer_path)
        # except OSError as e:
        #     if e.errno != os.errno.EEXIST:
        #         raise
        torch.set_default_dtype(torch.double)

        self.buffer_path = buffer_path
        self.smoothing_factor = smoothing_factor
        self.stability_factor = stability_factor
        self.loss_function = loss_function
        self.loss_traj = []
        self.autoencoder_loss_traj = []

        ############
        # SAMPLERS #
        ############

        self.data_sampler = data_sampler

        self.normalization_factor = self.data_sampler.normalization_factor
        self.data_dim = self.data_sampler.data_dim

        self.input_network_dim = self.data_sampler.data["X"].shape[1]
        self.output_network_dim = self.data_sampler.data["Y"].shape[1]

        ######################
        # NEURAL NET MODULES #
        ######################

        if load_model_from_buffer is True:  # restore the saved model
            if self.input_network_dim < self.data_dim:
                self.h, self.A, self.B = self.load()
            else:
                self.h = self.load()

        else:  # create fresh modules to be trained
            self.h = model.h(
                input_network_dim=self.input_network_dim,
                output_network_dim=self.output_network_dim,
                sirens=sirens,
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                first_omega_0=first_omega_0,
                hidden_omega_0=hidden_omega_0,
            )
            if self.input_network_dim < self.data_dim:
                self.A = model.A(network_dim=self.input_network_dim, data_dim=self.data_dim)
                self.B = model.B(network_dim=self.input_network_dim, data_dim=self.data_dim)

        if self.input_network_dim < self.data_dim:
            self.main_flow_func = model.MainFlow(self.h, self.A, self.B, enforce_positivity)
        else:
            self.main_flow_func = model.MainFlow(self.h, enforce_positivity=enforce_positivity)

        super().__init__(**kwargs)

        #########################################################
        # DEFINE HIGH-TO-LOW AND LOW-TO-HIGH INTERMEDIATE FLOWS #
        #########################################################

    def high2low(self, high_batch):
        if self.input_network_dim < self.data_dim:
            return self.A.forward(high_batch)
        else:
            return high_batch

    def low2high(self, low_batch):
        if self.input_network_dim < self.data_dim:
            return self.B.forward(low_batch)
        else:
            return low_batch

    def predict(self, input_x=None, to_numpy=True):
        input_x = self.data_sampler.data["X"] if input_x is None else input_x

        res = self.main_flow_func.forward(t=None, x=torch.tensor(input_x).double()).detach() / self.normalization_factor

        return res.numpy() if to_numpy else res

    def train(
        self,
        max_iter: int,
        data_batch_size: int,
        autoencoder_batch_size: int,
        data_lr: float,
        autoencoder_lr: float,
        sample_fraction: float = 1,
        iter_per_sample_update: Union[int, None] = None,
    ):
        """The training method for the DeepInterpolation model object

        Args:
            max_iter: The maximum iteration the network will be trained.
            data_batch_size: The size of the data sample batches to be generated in each iteration.
            autoencoder_batch_size: The size of the auto-encoder training batches to be generated in each iteration.
                Must be no greater than batch_size. .
            data_lr: The learning rate for network training.
            autoencoder_lr: The learning rate for network training the auto-encoder. Will have no effect if network_dim
                equal data_dim.
            sample_fraction: The best sample fraction to be filtered out of the velocity samples.
            iter_per_sample_update: The frequency of updating the subset of best samples (in terms of per iterations).
                Will have no effect if velocity_sample_fraction and time_course_sample_fraction are set to 1.
        """

        ############################
        ## SETTING THE OPTIMIZERS ##
        ############################

        # The optimizers for Neural Nets

        if self.input_network_dim < self.data_dim:

            self.optimizer = optim.Adam(
                list(self.A.parameters()) + list(self.h.parameters()) + list(self.B.parameters()),
                lr=data_lr,
                betas=(0.5, 0.9),
                weight_decay=2.5e-5,
                amsgrad=True,
            )

            self.ae_optimizer = optim.Adam(
                list(self.A.parameters()) + list(self.B.parameters()),
                lr=autoencoder_lr,
                betas=(0.5, 0.9),
                weight_decay=2.5e-5,
                amsgrad=True,
            )

        else:
            self.optimizer = optim.Adam(
                self.h.parameters(), lr=data_lr, betas=(0.5, 0.9), weight_decay=2.5e-5, amsgrad=True
            )

        ##############
        ## TRAINING ##
        ##############

        start_time = time.time()

        # Start with all the samples included
        sample_subset_indx = "all"

        # LET'S TRAIN!!
        for iter in range(max_iter):

            ###############################
            ### MAIN FLOW PASS ###
            ###############################

            if self.data_sampler is not None:

                # Set the gradients to zero
                self.h.zero_grad()
                if self.input_network_dim < self.data_dim:
                    self.A.zero_grad(), self.B.zero_grad()

                # Generate Data Batches
                indx, input_batch, output_batch, weight = self.data_sampler.generate_batch(
                    batch_size=data_batch_size, sample_subset_indices=sample_subset_indx
                )
                input_batch_noised = input_batch + torch.tensor(normal(loc=0, scale=0.1, size=input_batch.shape))

                # Forward Pass over the main flow
                v_hat_batch = self.main_flow_func.forward(t=None, x=input_batch)

                # Forward Pass again, this time for smoothing
                v_hat_batch_noised = self.main_flow_func.forward(t=None, x=input_batch_noised)

                # Calculate the loss value
                loss_value = self.loss_function(v_hat_batch, output_batch, weight)

                if self.smoothing_factor is not None:  # Adding Lipschitz smoothness regularizer term
                    loss_value += self.smoothing_factor * torch.nn.functional.mse_loss(
                        v_hat_batch, v_hat_batch_noised, reduction="mean"
                    )

                if self.stability_factor is not None:  # Adding Lyapunov stability regularizer term
                    loss_value += self.stability_factor * torch.mean(
                        torch.sum(torch.mul(output_batch, v_hat_batch), dim=1)
                    )

                # Backward Pass over the main flow
                loss_value.backward()  # Compute the gradients, but don't apply them yet
                # Now take an optimization step over main flow's parameters and velocity X variable
                self.optimizer.step()

            else:
                loss_value = np.nan

            #########################
            ### AUTO-ENCODER PASS ###
            #########################

            if self.input_network_dim < self.data_dim:

                # Set the gradients to zero
                self.A.zero_grad(), self.B.zero_grad()

                # Generate Data Batches
                ae_input_batch = torch.empty(0)

                if self.data_sampler is not None:
                    ae_input_batch = torch.cat([ae_input_batch, input_batch, output_batch])

                if autoencoder_batch_size != "all":
                    ae_input_batch = ae_input_batch[
                        np.random.choice(range(ae_input_batch.shape[0]), size=autoencoder_batch_size, replace=False)
                    ]

                # Forward Pass over the auto-encoder
                ae_low_batch = self.A.forward(ae_input_batch)
                ae_input_hat_batch = self.B.forward(ae_low_batch)

                # Calculate the autoencoder's loss value
                autoencoder_loss_value = torch.nn.functional.mse_loss(
                    ae_input_hat_batch, ae_input_batch, reduction="mean"
                )
                # Backward Pass over the auto-encoder
                autoencoder_loss_value.backward()
                self.ae_optimizer.step()

            else:
                autoencoder_loss_value = np.nan

            self.loss_traj.append(loss_value)
            self.autoencoder_loss_traj.append(autoencoder_loss_value)

            ################################################################
            # Demonstrate the training progress and save the trained model #
            ################################################################

            # LET'S SAVE THE MODEL AFTER TRAINING
            if (iter + 1) % 100 == 0:
                print(
                    "Iter [%8d] Time [%5.4f] regression loss [%.4f] autoencoder loss [%.4f]"
                    % (iter + 1, time.time() - start_time, loss_value, autoencoder_loss_value)
                )

                if self.input_network_dim < self.data_dim:
                    torch.save(self.A, self.buffer_path + "/A")
                    torch.save(self.B, self.buffer_path + "/B")
                torch.save(self.h, self.buffer_path + "/h")

                self.save()

                print("Model saved in path: %s" % self.buffer_path)

            #####################################
            # Update the subset of best samples #
            #####################################
            if sample_fraction < 1 and (iter + 1) % iter_per_sample_update == 0:
                # Taking the best subset of velocity samples
                _, _, v = self.data_sampler.generate_batch(batch_size="all", sample_subset_indices="all")
                v_hat = self.predict(input_batch)
                sample_subset_indx = subset_best_samples(sample_fraction, v_hat, v, self.loss_function)

    def save(self):
        if self.input_network_dim < self.data_dim:
            torch.save(self.A, self.buffer_path + "/A")
            torch.save(self.B, self.buffer_path + "/B")
        torch.save(self.h, self.buffer_path + "/h")

    def load(self):
        h = torch.load(self.buffer_path + "/h")
        if self.input_network_dim < self.data_dim:
            A = torch.load(self.buffer_path + "/A")
            B = torch.load(self.buffer_path + "/B")

            return h, A, B
        else:
            return h


def subset_best_samples(best_sample_fraction, y_hat, y, loss_func):
    assert y_hat.shape == y.shape, "The shape of the two arrays y_hat and y must be the same."
    diff = [loss_func(y_hat[i], y[i]) for i in range(y.shape[0])]

    return np.argsort(diff)[: int(best_sample_fraction * y.shape[0])]


class DataSampler(object):
    """This module loads and retains the data pairs (X, Y) and delivers the batches of them to the DeepInterpolation
    module upon calling. The module can load tha data from a .mat file. The file must contain two 2D matrices X and Y
    with equal rows.

    X: The spatial coordinates of each cell / binning / segmentation.
    Y: The expression values at the corresponding coordinates X.
    """

    def __init__(
        self,
        path_to_data: Union[str, None] = None,
        data: Union[AnnData, dict, None] = None,
        skey: str = "spatial",
        ekey: str = "M_s",
        wkey: Union[str, None] = None,
        normalize_data: bool = False,
        number_of_random_samples: str = "all",
        weighted: bool = False,
    ):
        """The initialization of the sampler for the expression data

        Args:
            path_to_data: A string that points to the file that stores the spatial coordinates and corresponding gene
                expression.
            data: If adata is an anndata.AnnData object, it must contain expression (embedding) / velocity information.
                If it is a dictionary, it must have the `X` and `Y` keys.
            skey: The key in .obsm that stores the spatial coordinates.
            ekey: adata.X or The key in .layers that stores the gene expression for each cell / segmentation / bin
            wkey: The key in .obs that stores the weight for each sample.
            normalize_data: Whether normalize the Y vectors upon loading by the data sampler.
            number_of_random_samples: The number of random samples loaded upfront from the box. If set to "all", all
                samples will be kept.
        """

        if path_to_data is None:
            weight = None
            self.data = {"X": None, "Y": None}

            if type(data) is anndata._core.anndata.AnnData:
                X, Y = data.obsm[skey], data.X if ekey == "X" else data.layers[ekey]
                if wkey:
                    weight = data.obs[wkey]

            elif type(data) is dict and ("X" in data.keys() and "Y" in data.keys()):
                X, Y = data["X"], data["Y"]

                if wkey in data.keys():
                    weight = data[wkey]
            else:
                raise ValueError(
                    f"data can be either an anndata object or a dictionary that has the `X` key and " f"the `Y` key"
                )

            good_ind = np.where(~np.isnan(Y.sum(1)))[0]
            X = X[good_ind, :]
            Y = Y[good_ind, :]

            self.data["X"], self.data["Y"] = X, Y

            if weight is not None:
                self.data["weight"] = weight
        else:
            self.data = sio.loadmat(path_to_data, appendmat=False)

        assert self.data["X"].shape[0] == self.data["Y"].shape[0], "The X and Y must have the same rows / cells."

        if number_of_random_samples == "all":  # If you choose to keep all the loaded samples
            self.X = self.data["X"]
            self.Y = self.data["Y"]
        else:  # Otherwise, it will keep a random subset of samples with the size given by "number_of_random_samples"
            indices = np.random.choice(range(self.data["X"].shape[0]), size=number_of_random_samples, repalce=False)
            self.X = self.data["X"][indices, :]
            self.Y = self.data["Y"][indices, :]

        self.train_size, self.data_dim = self.X.shape

        # Normalization factor for the data
        if normalize_data is True:
            self.normalization_factor = 1.0 / max(np.linalg.norm(self.Y, axis=1))
            self.Y *= self.normalization_factor
        else:
            self.normalization_factor = 1.0

        if weighted:
            self.weighted = True
            if "weight" in self.data.keys():
                self.weight = self.data["weight"] if number_of_random_samples == "all" else self.data["weight"][indices]
            else:
                raise ValueError(f"When `weighted` is set to be True, your data has to have the `weight` key!")
        else:
            self.weighted = False

    def generate_batch(self, batch_size: int, sample_subset_indices: str = "all"):
        """Generate random batches of the given size "batch_size" from the (X, Y) sample pairs.

        Args:
            batch_size: If the batch_size is set to "all",  all the samples will be returned.
            sample_subset_indices: This argument is used when you want to further subset the samples (based on the
                factors such as quality of the samples). If set to "all", it means it won't filter out samples based on
                their qualities.
        """

        if sample_subset_indices == "all":
            sample_subset_indices = range(self.train_size)

        if batch_size == "all":
            if self.weighted:
                return (
                    sample_subset_indices,
                    tensor(self.X[sample_subset_indices, :]).double(),
                    tensor(self.Y[sample_subset_indices, :]).double(),
                    self.weight[sample_subset_indices],
                )
            else:
                return (
                    sample_subset_indices,
                    tensor(self.X[sample_subset_indices, :]).double(),
                    tensor(self.Y[sample_subset_indices, :]).double(),
                )
        else:
            indices = np.random.choice(sample_subset_indices, size=batch_size, replace=False)
            if self.weighted:
                return (
                    indices,
                    tensor(self.X[indices, :]).double(),
                    tensor(self.Y[indices, :]).double(),
                    self.weight[indices],
                )
            else:
                return indices, tensor(self.X[indices, :]).double(), tensor(self.Y[indices, :]).double(), None
