import torch


def weighted_mean(x, weights):
    if weights is None:
        return torch.mean(x)
    else:
        return torch.sum(weights * x) / torch.sum(weights)


def weighted_mad():
    """Mean absolute difference (weighted)"""
    return lambda source, target, weights: weighted_mean(torch.abs(source - target), weights)


def weighted_mse():
    """Mean squared error (weighted)"""
    return lambda source, target, weights: weighted_mean(torch.norm(source - target, dim=1) ** 2, weights)


def weighted_cosine_distance():
    """Cosine similarity (weighted)"""
    return lambda source, target, weights: 1 - weighted_mean(
        torch.nn.functional.cosine_similarity(source, target), weights
    )


def mad():
    """Mean absolute difference"""
    return lambda source, target: torch.mean(torch.abs(source - target))  # nn.L1Loss()(input, target)


def mse():
    """Mean squared error"""
    return lambda source, target: torch.mean(torch.norm(source - target, dim=1) ** 2)  # nn.MSELoss()(input, target)


def cosine_distance():
    """Cosine similarity"""
    return lambda source, target: 1 - torch.mean(torch.nn.functional.cosine_similarity(source, target))
    # y = torch.FloatTensor([1]);
    # nn.CosineEmbeddingLoss()(input, target, y)
