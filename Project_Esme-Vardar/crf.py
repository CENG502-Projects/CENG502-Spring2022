from tkinter import BASELINE
import torch.nn as nn
import torch.nn.functional as F
import torch

def binary_comp_calc(all_data, all_label,  \
                batch_size, device, num_ways):
    
    binary_comp = torch.zeros((batch_size,all_data.size()[1],all_data.size()[1])).to(device)
    for i in range(binary_comp.size()[0]):
        for j in range(binary_comp.size()[1]):
            for k in range(binary_comp.size()[2]):
                t_sim = F.relu(F.cosine_similarity(all_data[i,j,:],all_data[i,k,:], dim=0))
                if (k > 4 or j > 4) and (k != j):
                    binary_comp[i,j,k] = (1.001 - t_sim) / (num_ways - 1) 
                else:
                    if (k > 4 or j > 4):
                        binary_comp[i,j,k] = t_sim + 0.001
                    else:
                        if all_label[i,j] == all_label[i,k]:
                            binary_comp[i,j,k] = t_sim + 0.001
                        else:
                            binary_comp[i,j,k] = (1.001 - t_sim) / (num_ways - 1) 

    return binary_comp



class CRF(nn.Module):
    def __init__(self, output_dim, lbp_count):
        super(CRF, self).__init__()

        self.lpb_count = lbp_count

        #self.crf_belief_transform = nn.Linear(128, output_dim)
        #nn.init.xavier_uniform(self.crf_belief_transform.weight)

    def init(self, inp_data, batch_size, x_dim, y_dim, unary_comp, binary_comp, num_supports, device, all_label):
        
        self.device = device
        self.message = torch.ones(batch_size,x_dim, x_dim, y_dim,self.lpb_count,requires_grad=True) / y_dim
        self.message = self.message.to(self.device)
        self.num_ways = y_dim
        self.unary_comp = unary_comp
        self.binary_comp = binary_comp
        self.num_supports = num_supports
        self.batch_size = batch_size
        
        #self.belief = F.softmax(self.crf_belief_transform(inp_data).to(self.device),dim=2)
        self.belief = self.get_initial_belief(inp_data, device, all_label, num_supports)
        self.belief = self.belief.unsqueeze(3).repeat(1,1,1,self.lpb_count)





    def get_initial_belief(self, inp_data, device, all_label, num_supports):
        belief = torch.zeros(self.batch_size, inp_data.size()[1], self.num_ways,requires_grad=True).to(device)
        templates = torch.zeros(self.batch_size, self.num_ways, inp_data.size()[2],requires_grad=True).to(device)

        for i in range(self.num_ways):
            templates[:,i,:] = inp_data[:,i,:]
        

        for j in range(belief.size()[1]):
            for k in range(belief.size()[2]):
                belief[:,j,k] = F.cosine_similarity(templates[:,k,:], inp_data[:,j,:],dim = 1)
        belief = F.softmax(belief,dim=2)
        return belief
        

    def loopy_belief_prop(self, affinity_mat):
        
        for i in range(self.lpb_count-1):
            for j in range(self.message.size()[1]):
                for k in range(self.message.size()[2]):
                    self.message[:,j,k,:,i+1] = F.normalize(self.binary_comp[:,j,k].unsqueeze(1) * \
                        (torch.div(self.belief[:,j,:,i], self.message[:,k,j,:,i])) + 0.0001,p=1, dim=1)

                    

            for j in range(self.belief.size()[1]):
                inter_msg = torch.ones(self.batch_size, self.belief.size()[2]).to(self.device)
                for k in range(self.message.size()[1]):
                    #Neighbor olayı şu an için yok nasıl yapılacağı şüpheli ?
                    #initial neighboring olabilir mi?
                    bl_mat = torch.gt(affinity_mat[:,j,k],0.001) 
                    inter_msg[bl_mat,:] += torch.mul(inter_msg[bl_mat],self.message[bl_mat,k,j,:,i+1])
                if j < self.num_supports:
                    inter_msg = inter_msg * self.unary_comp[:,j,:]
                self.belief[:,j,:,i+1] = inter_msg
            self.belief[:,:,:,i+1] = F.softmax(self.belief[:,:,:,i+1],dim=2)


            
    def get_affinity_mat(self, affinity_mat, bp_affinity_mat):
        it_affinity_mat = torch.zeros(affinity_mat.size(),requires_grad=True).to(self.device)
        ret_affinity_mat = torch.zeros(affinity_mat.size()).to(self.device)
        for i in range(it_affinity_mat.size()[1]):
            for j in range(it_affinity_mat.size()[2]):
                for k in range(self.num_ways):
                    it_affinity_mat[:,i,j] += self.belief[:,i,k,self.lpb_count-1] * self.belief[:,j,k,self.lpb_count-1]

        """    
        for i in range(it_affinity_mat.size()[1]):
            for j in range(it_affinity_mat.size()[2]):
                div_1 = torch.zeros(it_affinity_mat.size()[0]).to(self.device)
                div_2 = torch.zeros(it_affinity_mat.size()[0]).to(self.device)
                for k in range(affinity_mat.size()[2]):
                    div_1 += affinity_mat[:,i,j] * bp_affinity_mat[:,i,k]
                    div_2 += affinity_mat[:,i,j] 
                div_2 += 0.01
                div_1 += 0.01
                ret_affinity_mat[:,i,j] = torch.div(it_affinity_mat[:,i,j] * affinity_mat[:,i,j], \
                    torch.div(div_1,div_2))
        
        """

        return it_affinity_mat