import numpy as np
import torch
import torch.nn as nn

###########################################
# THE HIGH-TO-LOW-DIMENSIONAL TRANSFORMER #
###########################################


class A(nn.Module):
    def __init__(
        self,
        network_dim,
        data_dim,
        hidden_features=256,
        hidden_layers=1,
        activation_function=torch.nn.functional.leaky_relu,
    ):

        super(A, self).__init__()  # Call to the super-class is necessary

        self.f = activation_function
        self.name = "model/A"

        self.layer1 = nn.Linear(data_dim, hidden_features)

        self.net = []
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))

        self.hidden_layers = nn.Sequential(*self.net)
        self.outlayer = nn.Linear(256, network_dim)

        # torch.nn.init.normal_(self.layer1.weight, std=.02)
        # torch.nn.init.normal_(self.layer2.weight, std=.02)
        # torch.nn.init.normal_(self.layer3.weight, std=.02)

    def forward(self, inp):

        out = self.f(self.layer1(inp), negative_slope=0.2)
        out = self.f(self.hidden_layers(out), negative_slope=0.2)
        out = self.outlayer(out)
        return out


###########################################
# THE LOW-TO-HIGH-DIMENSIONAL TRANSFORMER #
###########################################


class B(nn.Module):
    def __init__(
        self,
        network_dim,
        data_dim,
        hidden_features=256,
        hidden_layers=3,
        activation_function=torch.nn.functional.leaky_relu,
    ):

        super(B, self).__init__()  # Call to the super-class is necessary

        self.f = activation_function
        self.name = "model/B"

        self.layer1 = nn.Linear(network_dim, hidden_features)

        self.net = []
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))

        self.hidden_layers = nn.Sequential(*self.net)
        self.outlayer = nn.Linear(hidden_features, data_dim)

        # torch.nn.init.normal_(self.layer1.weight, std=.02)
        # torch.nn.init.normal_(self.layer2.weight, std=.02)
        # torch.nn.init.normal_(self.layer3.weight, std=.02)

    def forward(self, inp):

        out = self.f(self.layer1(inp), negative_slope=0.2)
        out = self.f(self.hidden_layers(out), negative_slope=0.2)
        out = self.outlayer(out)
        return out


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
        input_network_dim,
        output_network_dim,
        hidden_features=256,
        hidden_layers=3,
        sirens=False,
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

        self.layer1 = nn.Linear(input_network_dim, hidden_features)
        if sirens:
            torch.nn.init.uniform_(self.layer1.weight, -1 / input_network_dim, 1 / input_network_dim)
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

        self.outlayer = nn.Linear(hidden_features, output_network_dim)
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
        e_low = self.h.forward(x_low)
        e_hat = self.B(e_low) if self.B is not None else e_low

        if freeze is not None:
            for i in freeze:
                if len(e_hat.shape) == 1:
                    e_hat[i] = 0
                elif len(e_hat.shape) == 2:
                    e_hat[:, i] = 0
                else:
                    raise ValueError("Invalid output data shape. Please debug.")

        # forcing the x to remain positive: set velocity to 0 if x<=0 and v<0
        if self.enforce_positivity:
            e_hat *= ~(e_hat < 0)  # ~((x <= 0) * (v_hat < 0))

        return e_hat
