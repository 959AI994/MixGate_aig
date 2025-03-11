from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Callable, List
import os.path as osp

import numpy as np 
import torch
import shutil
import os
import copy
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from mixgate.utils.data_utils import read_npz_file
from mixgate.utils.aiger_utils import aig_to_xdata
from mixgate.utils.circuit_utils import get_fanin_fanout, read_file, add_node_index, feature_gen_connect
from mixgate.utils.dataset_utils import *
from mixgate.utils.data_utils import construct_node_feature
from mixgate.utils.dag_utils import return_order_info

class NpzParser_Pair():
    '''
        Parse the npz file into an inmemory torch_geometric.data.Data object
    '''
    def __init__(self, data_dir, circuit_path, \
                 random_shuffle=True, trainval_split=0.9): 
        self.data_dir = data_dir
        dataset = self.inmemory_dataset(data_dir, circuit_path)
        if random_shuffle:
            perm = torch.randperm(len(dataset))
            dataset = dataset[perm]
        data_len = len(dataset)
        training_cutoff = int(data_len * trainval_split)
        self.train_dataset = dataset[:training_cutoff]
        self.val_dataset = dataset[training_cutoff:]
        # self.train_dataset = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        # self.val_dataset = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    def get_dataset(self):
        return self.train_dataset, self.val_dataset
    
    class inmemory_dataset(InMemoryDataset):
        def __init__(self, root, circuit_path, transform=None, pre_transform=None, pre_filter=None):
            self.name = 'npz_inmm_dataset'
            self.root = root
            self.circuit_path = circuit_path
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])
        
        @property
        def raw_dir(self):
            return self.root

        @property
        def processed_dir(self):
            name = 'inmemory'
            return osp.join(self.root, name)

        @property
        def raw_file_names(self) -> List[str]:
            return [self.circuit_path]

        @property
        def processed_file_names(self) -> str:
            return ['data.pt']

        def download(self):
            pass

        ### for aig
        def process(self):
            data_list = []
            tot_pairs = 0
            circuits = read_npz_file(self.circuit_path)['circuits'].item()
            j = 0
            for cir_idx, cir_name in enumerate(circuits):
                # if len(circuits[cir_name]) != 16:
                #     print(f"Skipping circuit {cir_name} with length {len(circuits[cir_name])}")
        
                #if len(circuits[cir_name]) == 17:
                print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx+1, len(circuits), (cir_idx+1) / len(circuits) *100))
                
                #aig:
                x = circuits[cir_name]["aig_x"]
                edge_index = circuits[cir_name]["aig_edge_index"]
                # is_pi = circuits[cir_name]["is_pi"]
                # no_edges = circuits[cir_name]["no_edges"]
                prob = circuits[cir_name]["aig_prob"]
                # backward_level = circuits[cir_name]["backward_level"]
                # forward_index = circuits[cir_name]["forward_index"]
                # forward_level = circuits[cir_name]["forward_level"]
                # no_nodes = circuits[cir_name]["no_nodes"]
                # backward_index = circuits[cir_name]["backward_index"]
                tt_dis = circuits[cir_name]["aig_tt_sim"]
                tt_pair_index = circuits[cir_name]["aig_tt_pair_index"]

                if len(tt_pair_index) == 0:
                    print('No tt : ', cir_name)
                    continue

                connect_label = None
                connect_pair_index = None


                graph = parse_pyg_mlpgate(
                    x, edge_index, tt_dis, tt_pair_index, 
                    prob, 
                )
                graph.num_nodes = len(x)
                graph.batch = torch.zeros(len(graph.x), dtype=torch.long)
                graph.gate = torch.tensor(circuits[cir_name]["aig_gate"], dtype=torch.float)

                #xmg:
                xmg_edge_index =  torch.tensor(circuits[cir_name]["xmg_edge_index"], dtype=torch.long).t().contiguous()
                xmg_x = circuits[cir_name]["xmg_x"]
                xmg_forward_level, xmg_forward_index, xmg_backward_level, xmg_backward_index = return_order_info(xmg_edge_index, torch.LongTensor(xmg_x).size(0))
                graph.xmg_tt_dis = circuits[cir_name]["xmg_tt_dis"]
                graph.xmg_tt_pair_index = torch.tensor(circuits[cir_name]["xmg_tt_pair_index"], dtype=torch.long).t().contiguous()
                if len(graph.xmg_tt_pair_index) == 0:
                    print('No tt : ', cir_name)
                    continue
                graph.xmg_x = torch.tensor(circuits[cir_name]["xmg_x"])
                graph.xmg_edge_index = torch.tensor(circuits[cir_name]["xmg_edge_index"], dtype=torch.long).t().contiguous()
                graph.xmg_prob = torch.tensor(circuits[cir_name]["xmg_prob"])
                graph.xmg_forward_level = torch.tensor(xmg_forward_level)
                graph.xmg_forward_index = torch.tensor(xmg_forward_index)
                graph.xmg_backward_level = torch.tensor(xmg_backward_level)
                graph.xmg_backward_index = torch.tensor(xmg_backward_index)
                graph.xmg_batch = torch.zeros(len(graph.xmg_x), dtype=torch.long)
                graph.xmg_gate = torch.tensor(xmg_x[:, 1:2], dtype=torch.float)

                #xag
                xag_edge_index = torch.tensor(circuits[cir_name]["xag_edge_index"], dtype=torch.long).t().contiguous()
                xag_x = circuits[cir_name]["xag_x"]
                xag_forward_level, xag_forward_index, xag_backward_level, xag_backward_index = return_order_info(xag_edge_index, torch.LongTensor(xag_x).size(0))
                graph.xag_tt_dis = circuits[cir_name]["xag_tt_dis"]
                graph.xag_tt_pair_index =  torch.tensor(circuits[cir_name]["xag_tt_pair_index"], dtype=torch.long).t().contiguous()
                if len(graph.xag_tt_pair_index) == 0:
                    print('No tt : ', cir_name)
                    continue
                graph.xag_x = torch.tensor(circuits[cir_name]["xag_x"])
                graph.xag_edge_index = torch.tensor(circuits[cir_name]["xag_edge_index"], dtype=torch.long).t().contiguous()
                graph.xag_prob = torch.tensor(circuits[cir_name]["xag_prob"])
                graph.xag_forward_level = torch.tensor(xag_forward_level)
                graph.xag_forward_index = torch.tensor(xag_forward_index)
                graph.xag_backward_level = torch.tensor(xag_backward_level)
                graph.xag_backward_index = torch.tensor(xag_backward_index)
                graph.xag_batch = torch.zeros(len(graph.xag_x), dtype=torch.long)
                graph.xag_gate = torch.tensor(circuits[cir_name]["xag_x"][:, 1:2], dtype=torch.float)

                #mig
                graph.mig_tt_dis = circuits[cir_name]["mig_tt_dis"]
                mig_x = circuits[cir_name]["mig_x"]
                mig_edge_index = torch.tensor(circuits[cir_name]["mig_edge_index"], dtype=torch.long).t().contiguous()
                graph.mig_tt_pair_index =  torch.tensor(circuits[cir_name]["mig_tt_pair_index"], dtype=torch.long).t().contiguous()
                graph.mig_x = torch.tensor(circuits[cir_name]["mig_x"])
                mig_forward_level, mig_forward_index, mig_backward_level, mig_backward_index = return_order_info(mig_edge_index, torch.LongTensor(mig_x).size(0))
                graph.mig_edge_index = torch.tensor(circuits[cir_name]["mig_edge_index"], dtype=torch.long).t().contiguous()
                if len(graph.mig_tt_pair_index) == 0:
                    print('No tt : ', cir_name)
                    continue
                graph.mig_prob = torch.tensor(circuits[cir_name]["mig_prob"])
                graph.mig_forward_index = mig_forward_index
                graph.mig_forward_level = mig_forward_level
                graph.mig_backward_index = mig_backward_index
                graph.mig_backward_level = mig_backward_level
                # graph.aig_gate = torch.tensor(circuits[cir_name]["aig_gate"])
                graph.mig_batch = torch.zeros(len(graph.mig_x), dtype=torch.long)
                graph.mig_gate = torch.tensor(circuits[cir_name]["mig_x"][:, 1:2], dtype=torch.float)
                graph.name = cir_name

                data_list.append(graph)
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
            print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))


        # ## for mig data
        # def process(self):
        #     data_list = []
        #     tot_pairs = 0
        #     circuits = read_npz_file(self.circuit_path)['circuits'].item()
        #     j = 0
        #     for cir_idx, cir_name in enumerate(circuits):
        #         # if len(circuits[cir_name]) != 16:
        #         #     print(f"Skipping circuit {cir_name} with length {len(circuits[cir_name])}")
        
        #         #if len(circuits[cir_name]) == 17:
        #         print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx+1, len(circuits), (cir_idx+1) / len(circuits) *100))
                
        #         #mig:
        #         x = circuits[cir_name]["mig_x"]
        #         edge_index = circuits[cir_name]["mig_edge_index"]
        #         # is_pi = circuits[cir_name]["is_pi"]
        #         # no_edges = circuits[cir_name]["no_edges"]
        #         prob = circuits[cir_name]["mig_prob"]
        #         # backward_level = circuits[cir_name]["backward_level"]
        #         # forward_index = circuits[cir_name]["forward_index"]
        #         # forward_level = circuits[cir_name]["forward_level"]
        #         # no_nodes = circuits[cir_name]["no_nodes"]
        #         # backward_index = circuits[cir_name]["backward_index"]
        #         tt_dis = circuits[cir_name]["mig_tt_dis"]
        #         tt_pair_index = circuits[cir_name]["mig_tt_pair_index"]

        #         if len(tt_pair_index) == 0:
        #             print('No tt : ', cir_name)
        #             continue

        #         connect_label = None
        #         connect_pair_index = None


        #         graph = parse_pyg_mlpgate(
        #             x, edge_index, tt_dis, tt_pair_index, 
        #             prob, 
        #         )
        #         graph.num_nodes = len(x)
        #         graph.batch = torch.zeros(len(graph.x), dtype=torch.long)

        #         #xmg:
        #         xmg_edge_index =  torch.tensor(circuits[cir_name]["xmg_edge_index"], dtype=torch.long).t().contiguous()
        #         xmg_x = circuits[cir_name]["xmg_x"]
        #         xmg_forward_level, xmg_forward_index, xmg_backward_level, xmg_backward_index = return_order_info(xmg_edge_index, torch.LongTensor(xmg_x).size(0))
        #         graph.xmg_tt_dis = circuits[cir_name]["xmg_tt_dis"]
        #         graph.xmg_tt_pair_index = torch.tensor(circuits[cir_name]["xmg_tt_pair_index"], dtype=torch.long).t().contiguous()
        #         if len(graph.xmg_tt_pair_index) == 0:
        #             print('No tt : ', cir_name)
        #             continue
        #         graph.xmg_x = torch.tensor(circuits[cir_name]["xmg_x"])
        #         graph.xmg_edge_index = torch.tensor(circuits[cir_name]["xmg_edge_index"], dtype=torch.long).t().contiguous()
        #         graph.xmg_prob = torch.tensor(circuits[cir_name]["xmg_prob"])
        #         graph.xmg_forward_level = torch.tensor(xmg_forward_level)
        #         graph.xmg_forward_index = torch.tensor(xmg_forward_index)
        #         graph.xmg_backward_level = torch.tensor(xmg_backward_level)
        #         graph.xmg_backward_index = torch.tensor(xmg_backward_index)
        #         graph.xmg_batch = torch.zeros(len(graph.xmg_x), dtype=torch.long)
        #         graph.xmg_gate = torch.tensor(xmg_x[:, 1:2], dtype=torch.float)

        #         #xag
        #         xag_edge_index = torch.tensor(circuits[cir_name]["xag_edge_index"], dtype=torch.long).t().contiguous()
        #         xag_x = circuits[cir_name]["xag_x"]
        #         xag_forward_level, xag_forward_index, xag_backward_level, xag_backward_index = return_order_info(xag_edge_index, torch.LongTensor(xag_x).size(0))
        #         graph.xag_tt_dis = circuits[cir_name]["xag_tt_dis"]
        #         graph.xag_tt_pair_index =  torch.tensor(circuits[cir_name]["xag_tt_pair_index"], dtype=torch.long).t().contiguous()
        #         if len(graph.xag_tt_pair_index) == 0:
        #             print('No tt : ', cir_name)
        #             continue
        #         graph.xag_x = torch.tensor(circuits[cir_name]["xag_x"])
        #         graph.xag_edge_index = torch.tensor(circuits[cir_name]["xag_edge_index"], dtype=torch.long).t().contiguous()
        #         graph.xag_prob = torch.tensor(circuits[cir_name]["xag_prob"])
        #         graph.xag_forward_level = torch.tensor(xag_forward_level)
        #         graph.xag_forward_index = torch.tensor(xag_forward_index)
        #         graph.xag_backward_level = torch.tensor(xag_backward_level)
        #         graph.xag_backward_index = torch.tensor(xag_backward_index)
        #         graph.xag_batch = torch.zeros(len(graph.xag_x), dtype=torch.long)
        #         graph.xag_gate = torch.tensor(circuits[cir_name]["xag_x"][:, 1:2], dtype=torch.float)

        #         #aig
        #         graph.aig_tt_sim = circuits[cir_name]["aig_tt_sim"]
        #         graph.aig_tt_pair_index =  torch.tensor(circuits[cir_name]["aig_tt_pair_index"], dtype=torch.long).contiguous()
        #         graph.aig_x = torch.tensor(circuits[cir_name]["aig_x"])
        #         graph.aig_edge_index = torch.tensor(circuits[cir_name]["aig_edge_index"], dtype=torch.long).contiguous()
        #         if len(graph.aig_tt_pair_index) == 0:
        #             print('No tt : ', cir_name)
        #             continue
        #         graph.aig_prob = torch.tensor(circuits[cir_name]["aig_prob"])
        #         graph.aig_forward_index = torch.tensor(circuits[cir_name]["aig_forward_index"])
        #         graph.aig_forward_level = torch.tensor(circuits[cir_name]["aig_forward_level"])
        #         graph.aig_backward_index = torch.tensor(circuits[cir_name]["aig_backward_index"])
        #         graph.aig_backward_level = torch.tensor(circuits[cir_name]["aig_backward_level"])
        #         # graph.aig_gate = torch.tensor(circuits[cir_name]["aig_gate"])
        #         graph.aig_batch = torch.zeros(len(graph.aig_x), dtype=torch.long)
        #         graph.aig_gate = torch.tensor(circuits[cir_name]["aig_gate"], dtype=torch.float)
        #         graph.name = cir_name

        #         data_list.append(graph)
        #         #print("data_list =", len(data_list))
        #     # while j < len(data_list):
        #     #     for i in range(len(data_list)):
        #     #         current_batch = data_list[:i+1]
        #     #         try:
        #     #             data, slices = self.collate(current_batch)
        #     #             #print(f"Batch {i} processed successfully.")
        #     #             #print("data_list =", data_list[i])
        #     #             j += 1
        #     #         except Exception as e:
        #     #             print(f"Error processing Batch {i}: {e}")
        #     #             print("data_list =", data_list[i])
        #     #             del data_list[i]
        #     #             break
        #     data, slices = self.collate(data_list)
        #     torch.save((data, slices), self.processed_paths[0])
        #     print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
        #     print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))

        def __repr__(self) -> str:
            return f'{self.name}({len(self)})'

class AigParser():
    def __init__(self):
        pass
    
    def read_aiger(self, aig_path):
        circuit_name = os.path.basename(aig_path).split('.')[0]
        # tmp_aag_path = os.path.join(self.tmp_dir, '{}.aag'.format(circuit_name))
        x_data, edge_index = aig_to_xdata(aig_path)
        # os.remove(tmp_aag_path)
        # Construct graph object 
        x_data = np.array(x_data)
        edge_index = np.array(edge_index)
        tt_dis = []
        tt_pair_index = []
        prob = [0] * len(x_data)
        rc_pair_index = []
        is_rc = []
        graph = parse_pyg_mlpgate(
            x_data, edge_index, tt_dis, tt_pair_index, prob, rc_pair_index, is_rc
        )
        graph.name = circuit_name
        graph.PIs = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] != 0)]
        graph.POs = graph.backward_index[(graph['backward_level'] == 0) & (graph['forward_level'] != 0)]
        graph.no_connect = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] == 0)]
        
        return graph        
        
class BenchParser():
    def __init__(self, gate_to_index={'INPUT': 0, 'MAJ': 1, 'NOT': 2, 'AND': 3, 'OR': 4, 'XOR': 5}):
        self.gate_to_index = gate_to_index
        pass
    
    def read_bench(self, bench_path):
        circuit_name = os.path.basename(bench_path).split('.')[0]
        x_data = read_file(bench_path)
        x_data, num_nodes, _ = add_node_index(x_data)
        x_data, edge_index = feature_gen_connect(x_data, self.gate_to_index)
        for idx in range(len(x_data)):
            x_data[idx] = [idx, int(x_data[idx][1])]
        # os.remove(tmp_aag_path)
        # Construct graph object 
        x_data = np.array(x_data)
        edge_index = np.array(edge_index)
        tt_dis = []
        tt_pair_index = []
        prob = [0] * len(x_data)
        rc_pair_index = []
        is_rc = []
        graph = parse_pyg_mlpgate(
            x_data, edge_index, tt_dis, tt_pair_index, prob, rc_pair_index, is_rc
        )
        graph.name = circuit_name
        graph.PIs = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] != 0)]
        graph.POs = graph.backward_index[(graph['backward_level'] == 0) & (graph['forward_level'] != 0)]
        graph.no_connect = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] == 0)]
        
        return graph       
