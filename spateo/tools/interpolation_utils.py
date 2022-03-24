"""
Todo:
    * @Xiaojieqiu: update with Google style documentation, function typings, tests
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from numpy.random import normal
from torch.autograd import Variable

torch.manual_seed(0)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class SineLayer(nn.Module):
    # from https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=uTQfrFvah3Zc
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    """
    As discussed above, we aim to provide each sine nonlinearity with activations that are standard
    normal distributed, except in the case of the first layer, where we introduced a factor ω0 that increased
    the spatial frequency of the first layer to better match the frequency spectrum of the signal. However,
    we found that the training of SIREN can be accelerated by leveraging a factor ω0 in all layers of the
    SIREN, by factorizing the weight matrix W as W = Wˆ ∗ ω0, choosing.
    This keeps the distribution of activations constant, but boosts gradients to the weight matrix Wˆ by
    the factor ω0 while leaving gradients w.r.t. the input of the sine neuron unchanged
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class h(nn.Module):
    def __init__(
        self,
        network_dim,
        hidden_features=256,
        hidden_layers=3,
        sirens=True,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
    ):
        self.sirens, self.first_omega_0, self.hidden_omega_0 = (
            sirens,
            first_omega_0,
            hidden_omega_0,
        )

        super(h, self).__init__()  # Call to the super-class is necessary

        self.f = torch.sin if self.sirens else torch.nn.functional.leaky_relu
        self.name = "model/h"

        self.layer1 = nn.Linear(network_dim, hidden_features)
        if sirens:
            torch.nn.init.uniform_(self.layer1.weight, -1 / network_dim, 1 / network_dim)
        self.net = []
        for i in range(hidden_layers):
            if sirens:
                self.net.append(
                    SineLayer(
                        hidden_features,
                        hidden_features,
                        is_first=False,
                        omega_0=self.hidden_omega_0,
                    )
                )
            else:
                self.net.append(nn.Linear(hidden_features, hidden_features))

        self.hidden_layers = nn.Sequential(*self.net)

        self.outlayer = nn.Linear(hidden_features, network_dim)
        if sirens:
            torch.nn.init.uniform_(
                self.outlayer.weight,
                -np.sqrt(6 / hidden_features) / self.hidden_omega_0,
                np.sqrt(6 / hidden_features) / self.hidden_omega_0,
            )

    def forward(self, inp):

        out = (
            self.f(self.first_omega_0 * self.layer1(inp))
            if self.sirens
            else self.f(self.layer1(inp), negative_slope=0.2)
        )  # , negative_slope=0.2
        out = self.hidden_layers(out) if self.sirens else self.f(self.hidden_layers(out), negative_slope=0.2)  #
        out = self.outlayer(out)

        return out


class MainFlow(torch.nn.Module):
    def __init__(self, h, A=None, B=None, enforce_positivity=False):

        super(MainFlow, self).__init__()

        self.A = A
        self.B = B
        self.h = h
        self.enforce_positivity = enforce_positivity

    def forward(self, t, x, freeze=None):

        x_low = self.A(x) if self.A is not None else x
        v_low = self.h.forward(x_low)
        v_hat = self.B(v_low) if self.B is not None else v_low

        if freeze is not None:
            for i in freeze:
                if len(v_hat.shape) == 1:
                    v_hat[i] = 0
                elif len(v_hat.shape) == 2:
                    v_hat[:, i] = 0
                else:
                    raise ValueError("Invalid output data shape. Please debug.")

        # forcing the x to remain positive: set velocity to 0 if x<=0 and v<0
        if self.enforce_positivity:
            v_hat *= ~(v_hat < 0)  # ~((x <= 0) * (v_hat < 0))

        return v_hat


class deep_interpolation:
    def __init__(
        self,
        model,
        sirens,
        enforce_positivity,
        velocity_data_sampler,
        network_dim,
        velocity_loss_function,
        velocity_x_initialize,
        smoothing_factor,
        stability_factor,
        load_model_from_buffer,
        buffer_path,
        hidden_features=256,
        hidden_layers=3,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
        *args,
        **kwargs
    ):

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

        self.buffer_path = buffer_path
        self.network_dim = network_dim
        self.smoothing_factor = smoothing_factor
        self.stability_factor = stability_factor
        self.velocity_loss_function = velocity_loss_function
        self.velocity_loss_traj = []
        self.autoencoder_loss_traj = []

        ############
        # SAMPLERS #
        ############

        self.velocity_data_sampler = velocity_data_sampler

        self.velocity_normalization_factor = self.velocity_data_sampler.normalization_factor
        self.data_dim = self.velocity_data_sampler.data_dim

        assert network_dim <= self.data_dim, "Network dimension must be no greater than the data dimension."

        ###########################
        # VARIABLE INITIALIZATION #
        ###########################

        # Initialize X variable for velocity data
        if velocity_data_sampler is not None:
            if velocity_x_initialize == "load_from_buffer":
                self.velocity_x_variable = torch.load(self.buffer_path + "velocity_x_variable")
            else:
                self.velocity_x_variable = Variable(torch.tensor(velocity_x_initialize).double(), requires_grad=True)

        ######################
        # NEURAL NET MODULES #
        ######################

        if load_model_from_buffer is True:  # restore the saved model
            self.h = torch.load(self.buffer_path + "/h")
            if self.network_dim < self.data_dim:
                self.A = torch.load(self.buffer_path + "/A")
                self.B = torch.load(self.buffer_path + "/B")
        else:  # create fresh modules to be trained
            self.h = model.h(
                network_dim=network_dim,
                sirens=sirens,
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                first_omega_0=first_omega_0,
                hidden_omega_0=hidden_omega_0,
            )
            if self.network_dim < self.data_dim:
                self.A = model.A(network_dim=self.network_dim, data_dim=self.data_dim)
                self.B = model.B(network_dim=self.network_dim, data_dim=self.data_dim)

        if self.network_dim < self.data_dim:
            self.main_flow_func = model.MainFlow(self.h, self.A, self.B, enforce_positivity)
        else:
            self.main_flow_func = model.MainFlow(self.h, enforce_positivity=enforce_positivity)

        super().__init__(**kwargs)

        #########################################################
        # DEFINE HIGH-TO-LOW AND LOW-TO-HIGH INTERMEDIATE FLOWS #
        ####################################################################################################################

    def high2low(self, high_batch):
        if self.network_dim < self.data_dim:
            return self.A.forward(high_batch)
        else:
            return high_batch

    def low2high(self, low_batch):
        if self.network_dim < self.data_dim:
            return self.B.forward(low_batch)
        else:
            return low_batch

    def predict_expression(self, input_x=None, to_numpy=True):
        input_x = self.velocity_data_sampler.data["X"] if input_x is None else input_x

        res = (
            self.main_flow_func.forward(t=None, x=torch.tensor(input_x).double()).detach()
            / self.velocity_normalization_factor
        )

        return res.numpy() if to_numpy else res

    def train(
        self,
        max_iter,
        velocity_batch_size,
        autoencoder_batch_size,
        velocity_lr,
        velocity_x_lr,
        autoencoder_lr,
        velocity_sample_fraction=1,
        iter_per_sample_update=None,
    ):

        ############################
        ## SETTING THE OPTIMIZERS ##
        ############################

        # The optimizers for Neural Nets

        if self.network_dim < self.data_dim:

            self.velocity_optimizer = optim.Adam(
                list(self.A.parameters()) + list(self.h.parameters()) + list(self.B.parameters()),
                lr=velocity_lr,
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
            self.velocity_optimizer = optim.Adam(
                self.h.parameters(), lr=velocity_lr, betas=(0.5, 0.9), weight_decay=2.5e-5, amsgrad=True
            )

        # The optimizers of velocity X and time course X0 variables
        if hasattr(self, "velocity_x_variable"):
            self.velocity_x_optimizer = optim.SGD([self.velocity_x_variable], lr=velocity_x_lr)
            # Setting Previous Velocity loss for adaptive learning rate
            previous_velocity_loss = np.inf

        ##############
        ## TRAINING ##
        ##############

        start_time = time.time()

        # Start with all the samples included
        velocity_sample_subset_indx = "all"
        time_course_sample_subset_indx = "all"

        # LET'S TRAIN!!
        for iter in range(max_iter):

            ###############################
            ### MAIN FLOW VELOCITY PASS ###
            ###############################

            if self.velocity_data_sampler is not None:

                # Set the gradients to zero
                self.h.zero_grad()
                self.velocity_x_variable.grad = torch.zeros(self.velocity_x_variable.shape)
                if self.network_dim < self.data_dim:
                    self.A.zero_grad(), self.B.zero_grad()

                # Generate Data Batches
                indx, velocity_y_batch, v_batch, weight = self.velocity_data_sampler.generate_batch(
                    batch_size=velocity_batch_size, sample_subset_indx=velocity_sample_subset_indx
                )
                velocity_x_batch_noised = self.velocity_x_variable[indx] + torch.tensor(
                    normal(loc=0, scale=0.1, size=velocity_y_batch.shape)
                )

                # Forward Pass over the main flow
                v_hat_batch = self.main_flow_func.forward(t=None, x=self.velocity_x_variable[indx])

                # Forward Pass again, this time for smoothing
                v_hat_batch_noised = self.main_flow_func.forward(t=None, x=velocity_x_batch_noised)

                # Calculate the loss value
                weight_subset = None if weight is None else weight[indx]
                velocity_loss_value = self.velocity_loss_function(
                    v_hat_batch, v_batch, weight
                ) + self.velocity_loss_function(self.velocity_x_variable[indx], velocity_y_batch, weight_subset)

                if self.smoothing_factor is not None:  # Adding Lipschitz smoothness regularizer term
                    velocity_loss_value += self.smoothing_factor * torch.nn.functional.mse_loss(
                        v_hat_batch, v_hat_batch_noised, reduction="mean"
                    )

                if self.stability_factor is not None:  # Adding Lyapunov stability regularizer term
                    velocity_loss_value += self.stability_factor * torch.mean(
                        torch.sum(torch.mul(self.velocity_x_variable[indx], v_hat_batch), dim=1)
                    )

                # Backward Pass over the main flow
                velocity_loss_value.backward()  # Compute the gradients, but don't apply them yet
                # Now take an optimization step over main flow's parameters and velocity X variable
                self.velocity_optimizer.step()
                self.velocity_x_optimizer.step()

            else:
                velocity_loss_value = np.nan

            ###############################################
            ### MAIN FLOW TIME-COURSE PASS (NEURAL ODE) ###
            ###############################################

            #########################
            ### AUTO-ENCODER PASS ###
            #########################

            if self.network_dim < self.data_dim:

                # Set the gradients to zero
                self.A.zero_grad(), self.B.zero_grad()

                # Generate Data Batches
                ae_input_batch = torch.empty(0)

                if self.velocity_data_sampler is not None:
                    ae_input_batch = torch.cat([ae_input_batch, velocity_y_batch, v_batch])

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

            self.velocity_loss_traj.append(velocity_loss_value)
            self.autoencoder_loss_traj.append(autoencoder_loss_value)

            ################################################################
            # Demonstrate the training progress and save the trained model #
            ################################################################
            if (iter + 1) % 10 == 0:
                # Update the learning rate for velocity X
                if hasattr(self, "velocity_x_variable") and velocity_loss_value > previous_velocity_loss:
                    print("Saturation detected for velocity. Reducing the X learning rate...")
                    velocity_x_lr *= 0.1
                    self.velocity_x_optimizer = optim.SGD([self.velocity_x_variable], lr=velocity_x_lr)

                previous_velocity_loss = velocity_loss_value

            # LET'S SAVE THE MODEL AFTER TRAINING
            if (iter + 1) % 100 == 0:
                if self.network_dim < self.data_dim:
                    torch.save(self.A, self.buffer_path + "/A")
                    torch.save(self.B, self.buffer_path + "/B")
                torch.save(self.h, self.buffer_path + "/h")
                if hasattr(self, "time_course_x0_variable"):
                    torch.save(self.time_course_x0_variable, self.buffer_path + "/time_course_x0_variable")
                if hasattr(self, "velocity_x_variable"):
                    torch.save(self.velocity_x_variable, self.buffer_path + "/velocity_x_variable")

                print("Model saved in path: %s" % self.buffer_path)

            #####################################
            # Update the subset of best samples #
            #####################################
            if velocity_sample_fraction < 1 and (iter + 1) % iter_per_sample_update == 0:

                # Taking the best subset of velocity samples
                _, _, v = self.velocity_data_sampler.generate_batch(batch_size="all", sample_subset_indx="all")
                v_hat = self.predict_velocity(self.velocity_x_variable.data)
                velocity_sample_subset_indx = subset_best_samples(
                    velocity_sample_fraction, v_hat, v, self.velocity_loss_function
                )

    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path, map_location=device))
        # self.model.eval()


def subset_best_samples(best_sample_fraction, y_hat, y, loss_func):

    assert y_hat.shape == y.shape, "The shape of the two arrays y_hat and y must be the same."
    diff = [loss_func(y_hat[i], y[i]) for i in range(y.shape[0])]

    return np.argsort(diff)[: int(best_sample_fraction * y.shape[0])]
