import gpytorch
import torch
from tqdm import tqdm


def gp_train(model, likelihood, train_loader, train_epochs, method, N, device):
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

    epochs_iter = tqdm(range(train_epochs), desc="Epoch")
    for i in epochs_iter:
        if method == "SVGP":
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=True)
            for x_batch, y_batch in minibatch_iter:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                minibatch_iter.set_postfix(loss=loss.item())
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
