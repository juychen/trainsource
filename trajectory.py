import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy as sc


def trajectory(adata):
    sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
    #sc.logging.print_versions()
    sc.settings.set_figure_params(dpi=80, frameon=False, figsize=(3, 3), facecolor='white')  # low dpi (dots per inch) yields small inline figures
    #adata2=sc.read_h5ad("./GSE112274_Gef2020-12-27-17-56-31.h5ad")
    adata2=adata
    adata2.X=adata2.X.astype('float64')
    
    ##draw
    sc.tl.draw_graph(adata2)
    sc.pl.draw_graph(adata2, color='sens_preds')
    sc.pl.draw_graph(adata2, color='leiden')
    sc.pl.draw_graph(adata2, color='leiden_Pret')
    
    
    
    ## trajectory1
    sc.tl.paga(adata2, groups='leiden_Pret')
    sc.pl.paga(adata2, color=['leiden_Pret'])
    adata2.obs['leiden_Pret'].cat.categories
    sc.tl.paga(adata2, groups='leiden_Pret')
    sc.pl.paga(adata2, threshold=0.03, show=False)
    sc.tl.draw_graph(adata2, init_pos='paga')
    sc.pl.draw_graph(adata2, color=['leiden_Pret'], legend_loc='on data')
    pl.figure(figsize=(8, 2))
    for i in range(8):
        pl.scatter(i, 1, c=sc.pl.palettes.zeileis_28[i], s=200)
    pl.show()
    zeileis_colors = np.array(sc.pl.palettes.zeileis_28)
    sc.pl.paga_compare(
        adata2, threshold=0.03, title='', right_margin=0.2, size=10,
        edge_width_scale=0.5,legend_fontsize=12, fontsize=12, frameon=False,
        edges=True, save=True)
    adata2.uns['iroot'] = np.flatnonzero(adata2.obs['leiden_Pret']  == '0')[0]
    sc.tl.dpt(adata2)
    adata_raw = adata2
    sc.pp.log1p(adata_raw)
    sc.pp.scale(adata_raw)
    adata.raw = adata_raw
    sc.pl.draw_graph(adata2, color=['sens_preds', 'dpt_pseudotime'], legend_loc='on data',save=True)
    
    
    
    
    ## trajectory2
    sc.tl.paga(adata2, groups='leiden')
    sc.pl.paga(adata2, color=['leiden'])
    adata2.obs['leiden'].cat.categories
    sc.tl.paga(adata2, groups='leiden')
    sc.pl.paga(adata2, threshold=0.03, show=False)
    sc.tl.draw_graph(adata2, init_pos='paga')
    sc.pl.draw_graph(adata2, color=['leiden'], legend_loc='on data')
    pl.figure(figsize=(8, 2))
    for i in range(8):
        pl.scatter(i, 1, c=sc.pl.palettes.zeileis_28[i], s=200)
    pl.show()
    zeileis_colors = np.array(sc.pl.palettes.zeileis_28)
    sc.pl.paga_compare(
        adata2, threshold=0.03, title='', right_margin=0.2, size=10,
        edge_width_scale=0.5,legend_fontsize=12, fontsize=12, frameon=False,
        edges=True)
    adata2.uns['iroot'] = np.flatnonzero(adata2.obs['leiden']  == '0')[0]
    sc.tl.dpt(adata2)
    adata_raw = adata2
    sc.pp.log1p(adata_raw)
    sc.pp.scale(adata_raw)
    adata.raw = adata_raw
    sc.pl.draw_graph(adata2, color=['leiden', 'dpt_pseudotime'], legend_loc='on data',save=True)

    
    return adata2
    
