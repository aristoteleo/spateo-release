import scanpy as sc
import SpatialDE



adata=sc.read("test.h5ad")
counts = pd.DataFrame(adata.X.todense(), columns=adata.var_names, index=adata.obs_names)
coord = pd.DataFrame(adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index=adata.obs_names)
results = SpatialDE.run(coord, counts)
results.index = results["g"]

results = results.sort_values("qval")

results.to_csv(path_or_buf="SpatialDE.csv")
