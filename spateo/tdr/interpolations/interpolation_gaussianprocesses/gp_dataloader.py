from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse
from ....logging import logger_manager as lm

# class Dataset(torch.utils.data.Dataset):
#     def __init__(
#         self, 
#         adata: AnnData, 
#         keys: Union[str, list] = None,
#         spatial_key: str = "spatial",
#         batch_size: int = 1024, 
#         layer: str = "X",
#         shuffle=True, 
#         random_seed=42,
#     ):
#         adata = adata.copy()
#         adata.X = adata.X if layer == "X" else adata.layers[layer]
#         spatial_data = adata.obsm[spatial_key]
#         info_data = np.ones(shape=(spatial_data.shape[0], 1))
#         assert keys != None, "`keys` cannot be None."
#         keys = [keys] if isinstance(keys, str) else keys
#         obs_keys = [key for key in keys if key in adata.obs.keys()]
#         if len(obs_keys) != 0:
#             obs_data = np.asarray(adata.obs[obs_keys].values)
#             info_data = np.c_[info_data, obs_data]
#         var_keys = [key for key in keys if key in adata.var_names.tolist()]
#         if len(var_keys) != 0:
#             var_data = adata[:, var_keys].X
#             if issparse(var_data):
#                 var_data = var_data.A
#             info_data = np.c_[info_data, var_data]
#         info_data = info_data[:, 1:]
        
#         self.device = f"cuda:{device}" if torch.cuda.is_available() and device != "cpu" else "cpu"
#         torch.device(self.device)
        
#         self.train_x = torch.from_numpy(spatial_data).float()
#         self.train_y = torch.from_numpy(info_data).float()
#         if self.device == "cpu":
#             self.train_x = self.train_x.cpu()
#             self.train_y = self.train_y.cpu()
#         else:
#             self.train_x = self.train_x.cuda()
#             self.train_y = self.train_y.cuda()
#         self.nx = ot.backend.get_backend(self.train_x, self.train_y)
#         self.PCA_reduction = False
#         self.info_keys = {"obs_keys": obs_keys, "var_keys": var_keys}
#         train_dataset = TensorDataset(self.train_x, self.train_y)
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        
#     def __len__(self):
#         return len(train_loader)
    
#     def __getitem__(self, index):
            
def Dataset(
    adata: AnnData, 
    keys: Union[str, list] = None,
    spatial_key: str = "spatial",
    batch_size: int = 1024, 
    layer: str = "X",
    inducing_num: int = 512,
    shuffle=True, 
    random_seed=42,
    normalize_spatial: bool = True,
):
    adata = adata.copy()
    adata.X = adata.X if layer == "X" else adata.layers[layer]
    spatial_data = adata.obsm[spatial_key]
    info_data = np.ones(shape=(spatial_data.shape[0], 1))
    assert keys != None, "`keys` cannot be None."
    keys = [keys] if isinstance(keys, str) else keys
    obs_keys = [key for key in keys if key in adata.obs.keys()]
    if len(obs_keys) != 0:
        obs_data = np.asarray(adata.obs[obs_keys].values)
        info_data = np.c_[info_data, obs_data]
    var_keys = [key for key in keys if key in adata.var_names.tolist()]
    if len(var_keys) != 0:
        var_data = adata[:, var_keys].X
        if issparse(var_data):
            var_data = var_data.A
        info_data = np.c_[info_data, var_data]
    info_data = info_data[:, 1:]
    
    device = f"cuda:{device}" if torch.cuda.is_available() and device != "cpu" else "cpu"
    torch.device(device)
    
    train_x = torch.from_numpy(spatial_data).float()
    train_y = torch.from_numpy(info_data).float()
    if device == "cpu":
        train_x = train_x.cpu()
        train_y = train_y.cpu()
    else:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
    # TO-DO: add normalization and norm parameters
    if normalize_spatial:
        train_x, normalize_param = normalize_coords(train_x,device=device)
    else:
        normalize_param = None
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    inducing_idx = np.random.choice(train_x.shape[0], inducing_num) if train_x.shape[0] > inducing_num else np.arange(train_x.shape[0])
    inducing_points = train_x[inducing_idx,:].clone()
    return train_loader, inducing_points, normalize_param

def normalize_coords(
    data: torch.Tensor,
    device: str = 'cpu',
    normalize_param: Optional[dict] = None,
):
    if normalize_param is None:
        mean = data.mean(0)
    else:
        if device == "cpu":
            mean = normalize_param['mean'].cpu()
        else:
            mean = normalize_param['mean'].cuda()
    data = data - self.mean
    if normalize_param is None:
        variance = torch.sqrt(torch.sum(data**2))
    else:
        if device == "cpu":
            variance = normalize_param['variance'].cpu()
        else:
            variance = normalize_param['variance'].cuda()
    data = data / variance
    if mean.is_cuda():
        mean = mean.cpu()
    if variance.is_cuda():
        variance = variance.cpu()    
    normalize_param = {'mean':mean, 'variance':variance}
    return data, normalize_param
            
        
        
        
# def normalize_coords(self, data: Union[np.ndarray, torch.Tensor], given_normalize: bool = False):
#     if not given_normalize:
#         self.mean_data = _unsqueeze(self.nx)(self.nx.mean(data, axis=0), 0)
#     data = data - self.mean_data
#     if not given_normalize:
#         self.variance = self.nx.sqrt(self.nx.sum(data**2) / data.shape[0])
#     data = data / self.variance
#     return data
        
        
    