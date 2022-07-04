import torch.nn as nn
import torch.nn.functional as F
import torch
from crf import CRF
from gnn_layer import GraphConvolution
from crf import binary_comp_calc

class MCRFGNN(nn.Module):
    def __init__(self,num_ways, device):
        super(MCRFGNN, self).__init__()

        self.device = device
        self.num_ways = num_ways

        self.crf_module_1 = CRF(num_ways, 7)

        self.graph_conv_1 = GraphConvolution(128,64)
        self.graph_conv_2 = GraphConvolution(64,32)
        self.graph_conv_3 = GraphConvolution(32,16)


    def train(self):
        super(MCRFGNN,self).train()
        
        self.graph_conv_1.train()
        self.graph_conv_2.train()
        self.graph_conv_3.train()


    def eval(self):
        self.graph_conv_1.train(False)
        self.graph_conv_2.train(False)
        self.graph_conv_3.train(False)

    
    def init_crf(self, inp_data ,batch_size, x_dim, y_dim, unary_comp, binary_comp, num_supports, affinity_mat, crf_module, all_label):
        crf_module.init(inp_data, batch_size, x_dim, y_dim, unary_comp, binary_comp,num_supports, self.device, all_label)
        crf_module.loopy_belief_prop(affinity_mat)
        self.affinity_list.append(crf_module.get_affinity_mat(affinity_mat,self.bp_aff_mat))

    def get_crf_belief_list(self):
        return self.belief_list
    
    
    def forward(self, inp_data ,batch_size, x_dim, y_dim, unary_comp, binary_comp, num_supports, affinity_mat,all_data, all_label):
        self.affinity_list = []
        self.belief_list = []
        self.bp_aff_mat = affinity_mat
        self.affinity_mat = affinity_mat
        self.init_crf(inp_data ,batch_size, x_dim, y_dim, unary_comp, binary_comp, num_supports, affinity_mat, self.crf_module_1, all_label)
        inp_data = self.graph_conv_1(inp_data, self.affinity_mat) 
        binary_comp = binary_comp_calc(inp_data, all_label, \
        batch_size, self.device, self.num_ways)
        self.belief_list.append(self.crf_module_1.belief)

        
        self.init_crf(inp_data ,batch_size, x_dim, y_dim, unary_comp, binary_comp, num_supports, self.affinity_mat, self.crf_module_1, all_label)
        inp_data = self.graph_conv_2(inp_data, self.affinity_mat)
        binary_comp = binary_comp_calc(inp_data, all_label, \
        batch_size, self.device,self.num_ways)
        self.belief_list.append(self.crf_module_1.belief)
        
        self.init_crf(inp_data ,batch_size, x_dim, y_dim, unary_comp, binary_comp, num_supports, self.affinity_mat, self.crf_module_1, all_label)
        inp_data = self.graph_conv_3(inp_data, self.affinity_mat)
        binary_comp = binary_comp_calc(inp_data, all_label, \
        batch_size, self.device,self.num_ways)
        self.belief_list.append(self.crf_module_1.belief)


        belief_list = self.get_crf_belief_list()


        return self.affinity_mat,belief_list,self.affinity_list




