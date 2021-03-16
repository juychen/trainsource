import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy as sc


def trajectory(adata,now):
    #sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)
    #sc.logging.print_versions()
    #sc.settings.set_figure_params(dpi=80, frameon=False, figsize=(3, 3), facecolor='white')  # low dpi (dots per inch) yields small inline figures
    #adata=sc.read_h5ad("./GSE112274_Gef2020-12-27-17-56-31.h5ad")
    #adata=adata
    #adata.X=adata.X.astype('float64')
    
    ##draw
    sc.tl.draw_graph(adata)
    ##sc.pl.draw_graph(adata, color='sens_preds')
    #sc.pl.draw_graph(adata, color='leiden')
    #sc.pl.draw_graph(adata, color='leiden_trans')
    
    sc.pl.draw_graph(adata, color=['leiden','sens_label'], legend_loc='on data',save="Initial_graph_"+now, show=False)

    # Diffusion map graph
    sc.tl.diffmap(adata)
    sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_diffmap')
    sc.tl.draw_graph(adata)
    sc.pl.draw_graph(adata, color=['leiden','sens_label'], legend_loc='on data',save="Diffusion_graph_"+now, show=False)

    
    ## trajectory1
    sc.tl.paga(adata, groups='leiden',neighbors_key='Trans')
    sc.pl.paga(adata, color=['leiden'],save="Paga_initial"+now, show=False) 


    #Recomputing the embedding using PAGA-initialization
    sc.tl.draw_graph(adata, init_pos='paga')
    sc.pl.draw_graph(adata, color=['leiden'], legend_loc='on data',save="Paga_initialization_graph"+now, show=False)
    # pl.figure(figsize=(8, 2))
    # for i in range(8):
    #     pl.scatter(i, 1, c=sc.pl.palettes.zeileis_28[i], s=200)
    # pl.show()
    zeileis_colors = np.array(sc.pl.palettes.zeileis_28)
    sc.pl.paga_compare(
        adata, threshold=0.03, title='', right_margin=0.2, size=10,
        edge_width_scale=0.5,legend_fontsize=12, fontsize=12, frameon=False,
        edges=True,save="Paga_cp1", show=False)
    adata.uns['iroot'] = np.flatnonzero(adata.obs['leiden']  == '0')[0]
    sc.tl.dpt(adata)
    adata_raw = adata
    sc.pp.log1p(adata_raw)
    sc.pp.scale(adata_raw)
    adata.raw = adata_raw
    sc.pl.draw_graph(adata, color=['sens_preds', 'dpt_pseudotime'], legend_loc='on data',save="Pseudotime_graph"+now, show=False)
    
    return adata
    
