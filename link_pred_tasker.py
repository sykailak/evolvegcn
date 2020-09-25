# THIS IS NODE2VEC LINK PRED WITH IMPORTED DICTIONARY

import torch
import taskers_utils as tu
import utils as u

import pandas as pd

import numpy as np

import scipy.sparse as sp

import logging

import time
import random
import os


class Link_Pred_Tasker():

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

        # self.all_node_feats_dic = self.build_get_node_feats(args, dataset)  ##should be a dic
        file = os.path.join(args.sbm50_args['folder'], args.sbm50_args['dict_file'])
        read_dictionary = np.load(file, allow_pickle='TRUE').item()
        self.all_node_feats_dic = read_dictionary

        # delete later
        self.feats_per_node = 100

    def build_prepare_node_feats(self, args, dataset):
        if args.use_2_hot_node_feats or args.use_1_hot_node_feats:
            def prepare_node_feats(node_feats):
                return u.sparse_prepare_tensor(node_feats,
                                               torch_size=[dataset.num_nodes,
                                                           self.feats_per_node])
        else:
            prepare_node_feats = self.data.prepare_node_feats

        return prepare_node_feats

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

        if self.args.sbm50_args['dict_file']=='tennis_dict.npy':
            #print('sampling football data...')
            # Sampling label_adj
            num_sample = int(np.floor(len(label_adj['vals'])*0.02))
            indice = random.sample(range(len(label_adj['vals'])), num_sample)
            indice = torch.LongTensor(indice)
            label_adj['idx'] = label_adj['idx'][indice,:]
            label_adj['vals'] = label_adj['vals'][indice]
            #print('len(label_adj[vals]):',len(label_adj['vals']))

            # Sampling non_exisiting_adj
            num_sample = int(np.floor(len(non_exisiting_adj['vals'])*0.02))
            indice = random.sample(range(len(non_exisiting_adj['vals'])), num_sample)
            indice = torch.LongTensor(indice)
            non_exisiting_adj['idx'] = non_exisiting_adj['idx'][indice,:]
            non_exisiting_adj['vals'] = non_exisiting_adj['vals'][indice]
            #print('len(non_exisiting_adj[vals]):',len(non_exisiting_adj['vals']))

        label_adj['idx'] = torch.cat([label_adj['idx'], non_exisiting_adj['idx']])
        label_adj['vals'] = torch.cat([label_adj['vals'], non_exisiting_adj['vals']])
        return {'idx': idx,
                'hist_adj_list': hist_adj_list,
                'hist_ndFeats_list': hist_ndFeats_list,
                'label_sp': label_adj,
                'node_mask_list': hist_mask_list}







