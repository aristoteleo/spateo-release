def GM_lag_model(adata, 
                genes,
                group,
                n_neighbors=8,
                layer=None,
                ):
    group_num = adata.obs[group].value_counts()

    group_name = adata.obs[group]

    uniq_g, group_name = group_name.unique(), group_name.to_list()

    # Generate W from the GeoDataFrame
    w = weights.distance.KNN.from_array(E9_5.obsm['spatial'], k=n_neighbors)

    # Row-standardization
    w.transform = 'R'

    if genes is None:
        genes = adata.var.index[adata.var.use_for_pca]
    else:
        genes = adata.var.index.intersection(genes)

    for i in ['const'] + uniq_g + ['W_log_exp']:
        adata.var[str(i) + '_GM_lag_coeff'] = None
        adata.var[str(i) + '_GM_lag_zstat'] = None
        adata.var[str(i) + '_GM_lag_pval'] = None
        
    db = pd.DataFrame({'group': group_name})

    X = pd.get_dummies(data=db, drop_first=True)

    variable_names = X.columns.to_list()

    for i, cur_g in tqdm(enumerate(genes), desc="performing GM_lag_model and assign coefficient and p-val to cell type"):
        if layer is None:
            X['log_exp'] = adata[:, cur_g].X.A.flatten()
        else:
            X['log_exp'] = adata[:, cur_g].layers[layer].A.flatten()
        
        try:
            model = spreg.GM_Lag(X[['log_exp']].values, X[variable_names].values, 
                                 w=knn, name_y='log_exp', name_x=variable_names)
        except: 
            for ind, g in enumerate(['const'] + uniq_g + ['W_log_exp']): 
                adata.var.loc[cur_g, g + '_GM_lag_coeff'] = np.nan
                adata.var.loc[cur_g, g + '_GM_lag_zstat'] = np.nan
                adata.var.loc[cur_g, g + '_GM_lag_pval'] = np.nan
        finally:
            a = pd.DataFrame(model.betas, model.name_x + ['W_log_exp'], columns=['Coefficient'])

            b = pd.DataFrame(model.z_stat, model.name_x + ['W_log_exp'], columns=['z_stat', 'p_val'])

            df = a.merge(b, left_index=True, right_index=True)

            for ind, g in enumerate(['const'] + uniq_g + ['W_log_exp']): 
                adata.var.loc[cur_g, g + '_GM_lag_coeff'] = df.iloc[ind, 0]
                adata.var.loc[cur_g, g + '_GM_lag_zstat'] = df.iloc[ind, 1]
                adata.var.loc[cur_g, g + '_GM_lag_pval'] = df.iloc[ind, 2]
                
    return adata 
