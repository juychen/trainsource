import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy as sc


def trajectory(adata,now):
    sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)
    #sc.logging.print_versions()
    #sc.settings.set_figure_params(dpi=80, frameon=False, figsize=(3, 3), facecolor='white')  # low dpi (dots per inch) yields small inline figures
    #adata2=sc.read_h5ad("./GSE112274_Gef2020-12-27-17-56-31.h5ad")
    adata2=adata
    adata2.X=adata2.X.astype('float64')
    
    ##draw
    sc.tl.draw_graph(adata2)
    ##sc.pl.draw_graph(adata2, color='sens_preds')
    #sc.pl.draw_graph(adata2, color='leiden')
    #sc.pl.draw_graph(adata2, color='leiden_trans')
    
    
    
    ## trajectory1
    sc.tl.paga(adata2, groups='leiden_trans',neighbors_key='Trans')
    sc.pl.paga(adata2, color=['leiden_trans'], show=False)  ## bugs here ,

    sc.tl.draw_graph(adata2, init_pos='paga')
    sc.pl.draw_graph(adata2, color=['leiden_trans'], legend_loc='on data',save="Drawgraph_1"+now, show=False)
    # pl.figure(figsize=(8, 2))
    # for i in range(8):
    #     pl.scatter(i, 1, c=sc.pl.palettes.zeileis_28[i], s=200)
    # pl.show()
    zeileis_colors = np.array(sc.pl.palettes.zeileis_28)
    sc.pl.paga_compare(
        adata2, threshold=0.03, title='', right_margin=0.2, size=10,
        edge_width_scale=0.5,legend_fontsize=12, fontsize=12, frameon=False,
        edges=True,save="Paga_cp1", show=False)
    adata2.uns['iroot'] = np.flatnonzero(adata2.obs['leiden_trans']  == '0')[0]
    sc.tl.dpt(adata2)
    adata_raw = adata2
    sc.pp.log1p(adata_raw)
    sc.pp.scale(adata_raw)
    adata.raw = adata_raw
    sc.pl.draw_graph(adata2, color=['sens_preds', 'dpt_pseudotime'], legend_loc='on data',save="Drawgraph_2"+now, show=False)
    
    
    
    
    ## trajectory2
    sc.tl.paga(adata2, groups='leiden')
    sc.pl.paga(adata2, color='leiden',save="Paga_1"+now, show=False)
    adata2.obs['leiden'].cat.categories
    sc.tl.paga(adata2, groups='leiden')
    sc.pl.paga(adata2, threshold=0.03, show=False,save="Paga_3"+now, show=False)
    sc.tl.draw_graph(adata2, init_pos='paga')
    sc.pl.draw_graph(adata2, color=['leiden'], legend_loc='on data',save="Drawgraph_3"+now, show=False)
    pl.figure(figsize=(8, 2))
    # for i in range(8):
    #     pl.scatter(i, 1, c=sc.pl.palettes.zeileis_28[i], s=200)
    # pl.show()
    zeileis_colors = np.array(sc.pl.palettes.zeileis_28)
    sc.pl.paga_compare(
        adata2, threshold=0.03, title='', right_margin=0.2, size=10,
        edge_width_scale=0.5,legend_fontsize=12, fontsize=12, frameon=False,
        edges=True,save="Paga_cp2", show=False)
    adata2.uns['iroot'] = np.flatnonzero(adata2.obs['leiden']  == '0')[0]
    sc.tl.dpt(adata2)
    adata_raw = adata2
    sc.pp.log1p(adata_raw)
    sc.pp.scale(adata_raw)
    adata.raw = adata_raw
    sc.pl.draw_graph(adata2, color=['leiden', 'dpt_pseudotime'], legend_loc='on data',save="Drawgraph_4"+now, show=False)

    
    return adata2
    
