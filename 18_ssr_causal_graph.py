import pandas as pd
import numpy as np
import networkx as nx
from bs4 import BeautifulSoup
from pyvis.network import Network

# %% Constant.
lag = 96
dataset_name = 'default_15min'

# %% Load data.
p_height = pd.read_pickle(f'raw/6_{dataset_name}_p_{lag}.pkl')
p_mass = pd.read_pickle(f'raw/12_{dataset_name}_p_{lag}.pkl')

# %% Combine 'same_height' adjacency matrices.
def bulk_diag_one(*args):
    bulk = args[0]
    assert bulk.ndim == 2, "The dimension of input matrix should be 2."
    for a in args[1:]:
        assert a.ndim == 2, "The dimension of input matrix should be 2."
        bulk = np.block([
            [bulk, np.ones((bulk.shape[0], a.shape[1]))],
            [np.ones((a.shape[0], bulk.shape[1])), a]
        ])
    return bulk

col_names = []
height_colors = {
    'h80': '#263ba8',
    'h150': '#226103',
    'h320': '#b12f24'
}
color_names = []
p_height_sorted = []
for height, p_mat in p_height.items():
    col_names += [f'{height}.{x}' for x in p_mass.index]
    np.fill_diagonal(p_mat, 1)
    p_height_sorted.append(p_mat)
    color_names += [height_colors[height] for _ in p_mass.index]
p_all = bulk_diag_one(*p_height_sorted)
p_all = pd.DataFrame(p_all, index=col_names, columns=col_names)

# %% Fill in 'same_molecule' results.
p_mass_melted = pd.melt(p_mass, value_name='p', var_name=['cause', 'effect'], ignore_index=False)
for mass, row in p_mass_melted.iterrows():
    p_all.loc[f"{row['cause']}.{mass}", f"{row['effect']}.{mass}"] = row['p']

# %% Draw the causality graph
def draw(adjacency: np.ndarray, labels: list, colors: list, output_path: str):
    g = nx.DiGraph()
    g.add_edges_from(zip(*np.where(adjacency)))
    nx.set_node_attributes(g, {i: color for i, color in enumerate(colors)}, 'color')
    g = nx.relabel_nodes(g, {i: label for i, label in enumerate(labels)})
    h = Network('100vh', directed=True, cdn_resources='remote')
    h.from_nx(g)
    h = BeautifulSoup(h.generate_html(), 'html.parser')
    [s.extract() for s in h.find('head').find_all('center')]
    s = h.find('body').find('div', {'class': 'card'})
    s.replaceWith(s.find('div', {'id': 'mynetwork'}))
    with open(output_path, 'w') as f_:
        f_.write(str(h.prettify()))

draw(
    adjacency=p_all < 0.05,
    labels=col_names,
    colors=color_names,
    output_path=f'results/18_{dataset_name}_causal_graph_{lag}.html'
)
