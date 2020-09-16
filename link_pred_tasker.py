#THIS IS NODE2VEC LINK PRED VERSION I’M USING

import torch
import taskers_utils as tu
import utils as u

from stellargraph import StellarGraph
import pandas as pd

from stellargraph.data import BiasedRandomWalk

from gensim.models import Word2Vec
import numpy as np

import scipy.sparse as sp

import logging

logging.getLogger("gensim.models").setLevel(logging.WARNING)

import time


class Link_Pred_Tasker():
    '''
    Creates a tasker object which computes the required inputs for training on a link prediction
    task. It receives a dataset object which should have two attributes: nodes_feats and edges, this
    makes the tasker independent of the dataset being used (as long as mentioned attributes have the same
    structure).

    Based on the dataset it implements the get_sample function required by edge_cls_trainer.
    This is a dictionary with:
      - time_step: the time_step of the prediction
      - hist_adj_list: the input adjacency matrices until t, each element of the list
               is a sparse tensor with the current edges. For link_pred they're
               unweighted
      - nodes_feats_list: the input nodes for the GCN models, each element of the list is a tensor
                two dimmensions: node_idx and node_feats
      - label_adj: a sparse representation of the target edges. A dict with two keys: idx: M by 2
             matrix with the indices of the nodes conforming each edge, vals: 1 if the node exists
             , 0 if it doesn't

    There's a test difference in the behavior, on test (or development), the number of sampled non existing
    edges should be higher.
    '''

    def __init__(self, args, dataset):
        self.data = dataset
        # max_time for link pred should be one before
        self.max_time = dataset.max_time - 1
        self.args = args
        self.num_classes = 2

        if not (args.use_2_hot_node_feats or args.use_1_hot_node_feats):
            self.feats_per_node = dataset.feats_per_node

        # self.get_node_feats = self.build_get_node_feats(args,dataset)
        self.prepare_node_feats = self.build_prepare_node_feats(args, dataset)
        self.is_static = False

        self.all_node_feats_dic = self.build_get_node_feats(args, dataset)  ##should be a dic

    def build_prepare_node_feats(self, args, dataset):
        if args.use_2_hot_node_feats or args.use_1_hot_node_feats:
            def prepare_node_feats(node_feats):
                return u.sparse_prepare_tensor(node_feats,
                                               torch_size=[dataset.num_nodes,
                                                           self.feats_per_node])
        else:
            prepare_node_feats = self.data.prepare_node_feats

        return prepare_node_feats

    def build_get_node_feats(self, args, dataset):
        self.feats_per_node = 128

        def get_node_feats(adj):  # input is cur_adj
            # start measuring time
            # start = time.process_time()

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
            # print('str_walks:',str_walks)

            weighted_model = Word2Vec(str_walks, size=self.feats_per_node, window=5, min_count=0, sg=1, workers=1,
                                      iter=1)

            # Retrieve node embeddings and corresponding subjects
            node_ids = weighted_model.wv.index2word  # list of node IDs
            # change to integer
            for i in range(0, len(node_ids)):
                node_ids[i] = int(node_ids[i])

            weighted_node_embeddings = (
                weighted_model.wv.vectors)  # numpy.ndarray of size number of nodes times embeddings dimensionality

            # print('weighted_node_embeddings shape:',weighted_node_embeddings.shape)

            # print('node_ids:',node_ids)
            # print('weighted_node_embeddings:',weighted_node_embeddings)
            # print('node_ids[0]:',node_ids[0])
            # print('first weighted_model.wv:',weighted_model.wv[str(node_ids[0])])

            # create dic
            dic = dict(zip(node_ids, weighted_node_embeddings.tolist()))
            # ascending order
            dic = dict(sorted(dic.items()))

            # create matrix
            adj_mat = sp.lil_matrix((self.data.num_nodes, self.feats_per_node))

            for row_idx in node_ids:
                adj_mat[row_idx, :] = dic[row_idx]

            adj_mat = adj_mat.tocsr()
            adj_mat = adj_mat.tocoo()

            coords = np.vstack((adj_mat.row, adj_mat.col)).transpose()
            values = adj_mat.data

            row = list(coords[:, 0])
            col = list(coords[:, 1])

            indexx = torch.LongTensor([row, col])
            tensor_size = torch.Size([self.data.num_nodes, self.feats_per_node])

            degs_out = torch.sparse.FloatTensor(indexx, torch.FloatTensor(values), tensor_size)

            hot_1 = {'idx': degs_out._indices().t(), 'vals': degs_out._values()}

            # print(time.process_time() - start)

            return hot_1

        # create dic
        feats_dic = {}

        for i in range(self.data.max_time):
            print('current i to make embeddings:', i)
            cur_adj = tu.get_sp_adj(edges=self.data.edges,
                                    time=i,
                                    weighted=True,
                                    time_window=self.args.adj_mat_time_window)

            feats_dic[i] = get_node_feats(cur_adj)

        return feats_dic

    def get_sample(self, idx, test, **kwargs):
        hist_adj_list = []
        hist_ndFeats_list = []
        hist_mask_list = []
        existing_nodes = []
        for i in range(idx - self.args.num_hist_steps, idx + 1):
            cur_adj = tu.get_sp_adj(edges=self.data.edges,
                                    time=i,
                                    weighted=True,
                                    time_window=self.args.adj_mat_time_window)

            if self.args.smart_neg_sampling:
                existing_nodes.append(cur_adj['idx'].unique())
            else:
                existing_nodes = None

            node_mask = tu.get_node_mask(cur_adj, self.data.num_nodes)

            # node_feats = self.get_node_feats(cur_adj)

            node_feats = self.all_node_feats_dic[i]

            cur_adj = tu.normalize_adj(adj=cur_adj, num_nodes=self.data.num_nodes)

            hist_adj_list.append(cur_adj)
            hist_ndFeats_list.append(node_feats)
            hist_mask_list.append(node_mask)

        # This would be if we were training on all the edges in the time_window
        label_adj = tu.get_sp_adj(edges=self.data.edges,
                                  time=idx + 1,
                                  weighted=False,
                                  time_window=self.args.adj_mat_time_window)
        if test:
            neg_mult = self.args.negative_mult_test
        else:
            neg_mult = self.args.negative_mult_training

        if self.args.smart_neg_sampling:
            existing_nodes = torch.cat(existing_nodes)

        if 'all_edges' in kwargs.keys() and kwargs['all_edges'] == True:
            non_exisiting_adj = tu.get_all_non_existing_edges(adj=label_adj, tot_nodes=self.data.num_nodes)
        else:
            non_exisiting_adj = tu.get_non_existing_edges(adj=label_adj,
                                                          number=label_adj['vals'].size(0) * neg_mult,
                                                          tot_nodes=self.data.num_nodes,
                                                          smart_sampling=self.args.smart_neg_sampling,
                                                          existing_nodes=existing_nodes)

        # label_adj = tu.get_sp_adj_only_new(edges = self.data.edges,
        #                    weighted = False,
        #                    time = idx)

        label_adj['idx'] = torch.cat([label_adj['idx'], non_exisiting_adj['idx']])
        label_adj['vals'] = torch.cat([label_adj['vals'], non_exisiting_adj['vals']])
        return {'idx': idx,
                'hist_adj_list': hist_adj_list,
                'hist_ndFeats_list': hist_ndFeats_list,
                'label_sp': label_adj,
                'node_mask_list': hist_mask_list}






