
#imports
import torch
from stellargraph import StellarGraph
import pandas as pd
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
import numpy as np
import scipy.sparse as sp
import logging
logging.getLogger("gensim.models").setLevel(logging.WARNING)
import time
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--adj_mat_time_window',default=5, type=int)
parser.add_argument('--edge_filepath',default='/content/drive/My Drive/KindredEvolve/icehockey_1week_alladj.csv', type=argparse.FileType(mode='r'))
parser.add_argument('--save_dest', default='/content/drive/My Drive/KindredEvolve/dict.npy.csv', type=argparse.FileType(mode='r'))
args = parser.parse_args()



def get_node_feats(adj,feats_per_node,data):  # input is cur_adj

    edgelist = adj['idx'].cpu().data.numpy()
    source = edgelist[:, 0]
    target = edgelist[:, 1]
    weight = np.ones(len(source))

    G = pd.DataFrame({'source': source, 'target': target, 'weight': weight})
    G = StellarGraph(edges=G)
    rw = BiasedRandomWalk(G)

    weighted_walks = rw.run(
        nodes=list(G.nodes()),  # root nodes
        length=2,  # maximum length of a random walk
        n=5,  # number of random walks per root node
        p=1,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=0.5,  # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=True,  # for weighted random walks
        seed=42,  # random seed fixed for reproducibility
    )

    str_walks = [[str(n) for n in walk] for walk in weighted_walks]

    weighted_model = Word2Vec(str_walks, size=feats_per_node, window=5, min_count=0, sg=1, workers=1,
                              iter=1)

    # Retrieve node embeddings and corresponding subjects
    node_ids = weighted_model.wv.index2word  # list of node IDs
    # change to integer
    for i in range(0, len(node_ids)):
        node_ids[i] = int(node_ids[i])

    weighted_node_embeddings = (
        weighted_model.wv.vectors)  # numpy.ndarray of size number of nodes times embeddings dimensionality

    # create dic
    dic = dict(zip(node_ids, weighted_node_embeddings.tolist()))
    # ascending order
    dic = dict(sorted(dic.items()))

    # create matrix
    adj_mat = sp.lil_matrix((data.num_nodes, feats_per_node))

    for row_idx in node_ids:
        adj_mat[row_idx, :] = dic[row_idx]

    adj_mat = adj_mat.tocsr()
    adj_mat = adj_mat.tocoo()
    coords = np.vstack((adj_mat.row, adj_mat.col)).transpose()
    values = adj_mat.data

    row = list(coords[:, 0])
    col = list(coords[:, 1])


    all_ids = edges[:, [0, 1]] #0: FromNodeId, 1:ToNodeId
    num_nodes = all_ids.max() + 1

    indexx = torch.LongTensor([row, col])
    tensor_size = torch.Size([num_nodes, feats_per_node])

    degs_out = torch.sparse.FloatTensor(indexx, torch.FloatTensor(values), tensor_size)

    hot_1 = {'idx': degs_out._indices().t(), 'vals': degs_out._values()}

    return hot_1





def get_sp_adj(edges, time, weighted, time_window):
    class Namespace(object): #helps referencing object in a dictionary as dict.key instead of dict['key']
        def __init__(self, adict):
            self.__dict__.update(adict)
    ECOLS = Namespace({'source': 0,
                       'target': 1,
                       'time': 2,
                       'label': 3})  # --> added for edge_cls

    idx = edges['idx']
    subset = idx[:, ECOLS.time] <= time
    subset = subset * (idx[:, ECOLS.time] > (time - time_window))
    idx = edges['idx'][subset][:, [ECOLS.source, ECOLS.target]]
    vals = edges['vals'][subset]
    out = torch.sparse.FloatTensor(idx.t(), vals).coalesce()

    idx = out._indices().t()
    if weighted:
        vals = out._values()
    else:
        vals = torch.ones(idx.size(0), dtype=torch.long)

    return {'idx': idx, 'vals': vals}


def build_get_node_feats(args, edges):
    feats_per_node = 100

    # create dic
    feats_dic = {}

    for i in range(167):
        #print('current i to make embeddings:', i)
        cur_adj = get_sp_adj(edges=edges,
                                time=i,
                                weighted=True,
                                time_window=args.adj_mat_time_window)

        feats_dic[i] = get_node_feats(cur_adj, feats_per_node,data)

    return feats_dic


def load_edges(filepath, starting_line = 1):
    with open(filepath) as f:
        lines = f.read().splitlines()
    edges = [[float(r) for r in row.split(',')] for row in lines[starting_line:]]
    edges = torch.tensor(edges,dtype = torch.long)
    return edges


edges = load_edges(args.edge_filepath)
all_node_feats_dic = build_get_node_feats(args, edges)  ##should be a dic
np.save(args.save_dest, all_node_feats_dic)
