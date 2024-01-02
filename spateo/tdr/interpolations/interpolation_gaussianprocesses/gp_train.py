from .gp_dataloader import Dataset
from tqdm import tqdm

def gp_train(
    model,
    likelihood,
    train_loader,
    train_epochs,
):
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)
    
    epochs_iter = tqdm(range(train_epochs), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()