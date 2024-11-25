import gpytorch
import torch
from tqdm import tqdm

from spateo.alignment.utils import _iteration


def gp_train(model, likelihood, train_loader, train_epochs, method, N, device, keys, verbose=True):
    if torch.cuda.is_available() and device != "cpu":
        model = model.cuda()
        likelihood = likelihood.cuda()

    model.train()
    likelihood.train()
    # define the mll (loss)
    if method == "SVGP":
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N)
    else:
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=0.01,
    )

    progress_name = f"Interpolation based on Gaussian Process Regression for {keys[0]}"
    epochs_iter = _iteration(n=train_epochs, progress_name=progress_name, verbose=verbose)
    for i in epochs_iter:
        if method == "SVGP":
            # Within each iteration, we will go over each minibatch of data
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
        else:
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_loader["train_x"])
            # Calc loss and backprop gradients
            loss = -mll(output, train_loader["train_y"])
            loss.backward()
            optimizer.step()
