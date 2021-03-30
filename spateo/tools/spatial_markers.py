from tqdm import tqdm

def cluster_specific_genes(adata, 
                           group,
                           spatial_key='spatial',
                           genes=None,
                           layer=None,
                           n_neighbors=8,
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

        db = pandas.DataFrame(adata.obsm['spatial'], columns=['x', 'y'])

        adata.var['hotspot_num'], adata.var['hotspot_frac'], adata.var['hotspot_spec'] = None, None, None,
        adata.var['coldspot_num'], adata.var['coldspot_frac'], adata.var['coldspot_spec'] = None, None, None,
        adata.var['doughnut_num'], adata.var['doughnut_frac'], adata.var['doughnut_spec'] = None, None, None,
        adata.var['diamond_num'], adata.var['diamond_frac'], adata.var['diamond_spec'] = None, None, None,

        adata.var['hotspot_num_val'], adata.var['hotspot_frac_val'], adata.var['hotspot_spec_val'] = None, None, None,
        adata.var['coldspot_num_val'], adata.var['coldspot_frac_val'], adata.var['coldspot_spec_val'] = None, None, None,
        adata.var['doughnut_num_val'], adata.var['doughnut_frac_val'], adata.var['doughnut_spec_val'] = None, None, None,
        adata.var['diamond_num_val'], adata.var['diamond_frac_val'], adata.var['diamond_spec_val'] = None, None, None,

        hotspot_num, hotspot_frac, hotspot_spec = np.zeros(len(uniq_g)), np.zeros(len(uniq_g)), np.zeros(len(uniq_g))
        coldspot_num, coldspot_frac, coldspot_spec = np.zeros(len(uniq_g)), np.zeros(len(uniq_g)), np.zeros(len(uniq_g))
        doughnut_num, doughnut_frac, doughnut_spec = np.zeros(len(uniq_g)), np.zeros(len(uniq_g)), np.zeros(len(uniq_g))
        diamond_num, diamond_frac, diamond_spec = np.zeros(len(uniq_g)), np.zeros(len(uniq_g)), np.zeros(len(uniq_g))

        valid_inds_list = [np.array(group_name) == g for g in uniq_g]
        
        for i, cur_g in tqdm(enumerate(genes), desc="performing local Moran I analysis and assign genes and significant domaints to cell type"):
            if layer is None:
                db['exp'] = adata[:, cur_g].X.A.flatten()
            else:
                db['exp'] = adata[:, cur_g].layers[layer].A.flatten()

            db['w_exp'] = weights.spatial_lag.lag_spatial(w, db['exp'])

            lisa = esda.moran.Moran_Local(db['exp'], w)

            sig = 1 * (lisa.p_sim < 0.05)

            hotspot = 1 * (sig * lisa.q==1)
            coldspot = 3 * (sig * lisa.q==3)
            doughnut = 2 * (sig * lisa.q==2)
            diamond = 4 * (sig * lisa.q==4)

            for ind, g in enumerate(uniq_g):   
                valid_inds = valid_inds_list[ind]
                hotspot_num[ind], hotspot_frac[ind], hotspot_spec[ind] = sum(hotspot[valid_inds]), sum(hotspot[valid_inds]) / group_num[g], sum(hotspot[valid_inds]) / sum(hotspot)
                coldspot_num[ind], coldspot_frac[ind], coldspot_spec[ind] = sum(coldspot[valid_inds]), sum(coldspot[valid_inds]) / group_num[g], sum(coldspot[valid_inds]) / sum(coldspot)
                doughnut_num[ind], doughnut_frac[ind], doughnut_spec[ind] = sum(doughnut[valid_inds]), sum(doughnut[valid_inds]) / group_num[g], sum(doughnut[valid_inds]) / sum(doughnut)
                diamond_num[ind], diamond_frac[ind], diamond_spec[ind] = sum(diamond[valid_inds]), sum(diamond[valid_inds]) / group_num[g], sum(diamond[valid_inds]) / sum(diamond)

            adata.var.loc[cur_g, 'hotspot_num_val'], adata.var.loc[cur_g, 'hotspot_frac_val'], adata.var.loc[cur_g, 'hotspot_spec_val'] = np.max(hotspot_num), np.max(hotspot_frac), np.max(hotspot_spec)
            adata.var.loc[cur_g, 'coldspot_num_val'], adata.var.loc[cur_g, 'coldspot_frac_val'], adata.var.loc[cur_g, 'coldspot_spec_val'] = np.max(coldspot_num), np.max(coldspot_frac), np.max(coldspot_spec)
            adata.var.loc[cur_g, 'doughnut_num_val'], adata.var.loc[cur_g, 'doughnut_frac_val'], adata.var.loc[cur_g, 'doughnut_spec_val'] = np.max(doughnut_num), np.max(doughnut_frac), np.max(doughnut_spec)
            adata.var.loc[cur_g, 'diamond_num_val'], adata.var.loc[cur_g, 'diamond_frac_val'], adata.var.loc[cur_g, 'diamond_spec_val'] = np.max(diamond_num), np.max(diamond_frac), np.max(diamond_spec)
            
            adata.var.loc[cur_g, 'hotspot_num'], adata.var.loc[cur_g, 'hotspot_frac'], adata.var.loc[cur_g, 'hotspot_spec'] = uniq_g[np.argsort(hotspot_num)[-1]], uniq_g[np.argsort(hotspot_frac)[-1]], uniq_g[np.argsort(hotspot_spec)[-1]]
            adata.var.loc[cur_g, 'coldspot_num'], adata.var.loc[cur_g, 'coldspot_frac'], adata.var.loc[cur_g, 'coldspot_spec'] = uniq_g[np.argsort(coldspot_num)[-1]], uniq_g[np.argsort(coldspot_frac)[-1]], uniq_g[np.argsort(coldspot_spec)[-1]]
            adata.var.loc[cur_g, 'doughnut_num'], adata.var.loc[cur_g, 'doughnut_frac'], adata.var.loc[cur_g, 'doughnut_spec'] = uniq_g[np.argsort(doughnut_num)[-1]], uniq_g[np.argsort(doughnut_frac)[-1]], uniq_g[np.argsort(doughnut_spec)[-1]]
            adata.var.loc[cur_g, 'diamond_num'], adata.var.loc[cur_g, 'diamond_frac'], adata.var.loc[cur_g, 'diamond_spec'] = uniq_g[np.argsort(diamond_num)[-1]], uniq_g[np.argsort(diamond_frac)[-1]], uniq_g[np.argsort(diamond_spec)[-1]]

        return adata 
      
      
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
