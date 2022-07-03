import sys
import os
import signal
import time
from time import perf_counter

import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as nn_func
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision.transforms.functional as xform

class SMDSwBdLSTMDataset(data_utils.Dataset):
    def __init__(self, root_path):
        super(SMDSwBdLSTMDataset, self).__init__()
        self.root_path = root_path
        self.preprocessed_data_path = [fp for fp in os.listdir(self.root_path) if any(fp.endswith(ext) for ext in ['h5'])]

    def __len__(self):
        return len(self.preprocessed_data_path)

    def __getitem__(self, index):
        preprocessed_data = h5py.File(os.path.join(self.root_path, self.preprocessed_data_path[index]))
        assert('number_of_poses_in_sequence' in preprocessed_data.keys())   # NOTE(ff-k): Sequence-dependent parameter
        assert('vertex_count_of_mesh' in preprocessed_data.keys())          # NOTE(ff-k): # of vertices for the mesh. Note that this should be the same for all meshes in the whole dataset
        assert('feature_vector_length' in preprocessed_data.keys())         # NOTE(ff-k): 9 for ACAP
        assert('feature_vectors' in preprocessed_data.keys())               # NOTE(ff-k): size = (number_of_poses_in_sequence, vertex_count_of_mesh, feature_vector_length)
        assert('neighbour_feature_vectors' in preprocessed_data.keys())     # NOTE(ff-k): size = (number_of_poses_in_sequence, vertex_count_of_mesh, feature_vector_length)
        
        # NOTE(ff-k): We expect input data to be normalized and these are the normalization parameters that should be included in the preprocessed data
        assert('feature_clamp_max' in preprocessed_data.keys())             # NOTE(ff-k):  0.95 in the original paper
        assert('feature_clamp_min' in preprocessed_data.keys())             # NOTE(ff-k): -0.95 in the original paper
        assert('feature_data_max' in preprocessed_data.keys())              # NOTE(ff-k): maximum element in feature vector
        assert('feature_data_min' in preprocessed_data.keys())              # NOTE(ff-k): maximum element in feature vector

        assert('feature_vectors' in preprocessed_data.keys())
        self_feature_vectors = preprocessed_data['feature_vectors']
        neighbour_feature_vectors = preprocessed_data['neighbour_feature_vectors'] # TODO(ff-k): Remove this from preprocessed data as we need to compute it in conv layer. Also, add degree per vertex, neighbour vertices etc.
        assert(self_feature_vectors.shape[0] == preprocessed_data['number_of_poses_in_sequence'])
        assert(self_feature_vectors.shape[1] == preprocessed_data['vertex_count_of_mesh'])
        assert(self_feature_vectors.shape[2] == preprocessed_data['feature_vector_length'])
        assert(self_feature_vectors.shape[0] == neighbour_feature_vectors.shape[0])
        assert(self_feature_vectors.shape[1] == neighbour_feature_vectors.shape[1])
        assert(self_feature_vectors.shape[2] == neighbour_feature_vectors.shape[2])

        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')

        tensor_self_feature_vectors = xform.to_tensor(self_feature_vectors).to(device=device, dtype=torch.float32)
        tensor_neighbour_feature_vectors = xform.to_tensor(neighbour_feature_vectors).to(device=device, dtype=torch.float32)

        return tensor_self_feature_vectors, tensor_neighbour_feature_vectors

class SMDSwBdLSTMConvLayer(nn.Module):
    def __init__(self, w1, w2):
        super(SMDSwBdLSTMConvLayer, self).__init__()
        self.w1 = w1
        self.w2 = w2
        # TODO(ff-k): bias
    def forward(self, self_feature_vectors, neighbour_feature_vectors):
        # NOTE(ff-k): See Section 3.2, Equation 2
        y_i = torch.tensordot(self_feature_vectors, self.w1, dims=([1], [0])) + torch.tensordot(neighbour_feature_vectors, self.w2, dims=([1], [0]))
        return torch.tanh(y_i)

class SMDSwBdLSTMSubNet(nn.Module):
    def __init__(self, feature_vector_length, vertex_count_of_mesh, lstm_num_layers, lstm_hidden_dimensions):
        super(SMDSwBdLSTMSubNet, self).__init__()

        self.feature_vector_length = feature_vector_length
        self.vertex_count_of_mesh = vertex_count_of_mesh
        lstm_io_size = self.vertex_count_of_mesh * self.feature_vector_length

        # TODO(ff-k): add bias terms after running with dummy data and properly handling batch size
        conv_outer_w1 = torch.empty(feature_vector_length, feature_vector_length)
        conv_outer_w2 = torch.empty(feature_vector_length, feature_vector_length)
        conv_middle_w1 = torch.empty(feature_vector_length, feature_vector_length)
        conv_middle_w2 = torch.empty(feature_vector_length, feature_vector_length)
        conv_inner_w1 = torch.empty(feature_vector_length, feature_vector_length)
        conv_inner_w2 = torch.empty(feature_vector_length, feature_vector_length)

        nn.init.xavier_uniform_(conv_outer_w1, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(conv_outer_w2, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(conv_middle_w1, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(conv_middle_w2, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(conv_inner_w1, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(conv_inner_w2, gain=nn.init.calculate_gain('tanh'))
        
        self.conv_outer = SMDSwBdLSTMConvLayer(nn.Parameter(conv_outer_w1), nn.Parameter(conv_outer_w2))
        self.conv_middle = SMDSwBdLSTMConvLayer(nn.Parameter(conv_middle_w1), nn.Parameter(conv_middle_w2))
        self.conv_inner = SMDSwBdLSTMConvLayer(nn.Parameter(conv_inner_w1), nn.Parameter(conv_inner_w2))
        self.cell = nn.LSTM(lstm_io_size, lstm_hidden_dimensions, lstm_num_layers, True, False, 0, False, lstm_io_size)
        self.tconv_inner = SMDSwBdLSTMConvLayer(nn.Parameter(torch.transpose(conv_inner_w1, 0, 1)), nn.Parameter(torch.transpose(conv_inner_w2, 0, 1)))
        self.tconv_middle = SMDSwBdLSTMConvLayer(nn.Parameter(torch.transpose(conv_middle_w1, 0, 1)), nn.Parameter(torch.transpose(conv_middle_w2, 0, 1)))
        self.tconv_outer = SMDSwBdLSTMConvLayer(nn.Parameter(torch.transpose(conv_outer_w1, 0, 1)), nn.Parameter(torch.transpose(conv_outer_w2, 0, 1)))

    def forward(self, self_feature_vectors, neighbour_feature_vectors, initial_states_h, initial_states_c):
        assert(self_feature_vectors.shape[1] == self.feature_vector_length)

        # TODO(ff-k): Precomputing neighbour values will not serve our purposes here. We have to do it in SMDSwBdLSTMConvLayer every single time :sad_parrot:
        val = self.conv_outer(self_feature_vectors, neighbour_feature_vectors)
        val = self.conv_middle(val, neighbour_feature_vectors)
        val = self.conv_inner(val, neighbour_feature_vectors)
        val, (initial_states_h_next, initial_states_c_next) = self.cell(val, (initial_states_h, initial_states_c))
        val = self.conv_inner(val, neighbour_feature_vectors)
        val = self.conv_middle(val, neighbour_feature_vectors)
        val = self.conv_outer(val, neighbour_feature_vectors)

        return self_feature_vectors + val, (initial_states_h_next, initial_states_c_next)

class SMDSwBdLSTMNet(nn.Module):
    def __init__(self, feature_vector_length, vertex_count_of_mesh, lstm_num_layers, lstm_hidden_dimensions, n_frames):
        super(SMDSwBdLSTMNet, self).__init__()

        self.feature_vector_length = feature_vector_length
        self.vertex_count_of_mesh = vertex_count_of_mesh
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_dimensions = lstm_hidden_dimensions

        self.n_frames = n_frames
        self.s_i = SMDSwBdLSTMSubNet(feature_vector_length, vertex_count_of_mesh, lstm_num_layers, lstm_hidden_dimensions)

    def forward(self, self_feature_vectors, neighbour_feature_vectors, h, c, reverse_direction):
        # TODO(ff-k): Handle reverse_direction
        Xs = []
        Xs.append(self_feature_vectors)
        for i in range(self.n_frames):
            X, (h_next, c_next) = self.s_i(Xs[len(Xs)-1], neighbour_feature_vectors, h, c)
            Xs.append(X)
            h = h_next
            c = c_next
        return Xs

def interrupt_handler(signum, frame):
    print('Received an interrupt signal and will terminate after completing the active epoch')
    globals()['interruped_by_user'] = True

def get_run_config(checkpoint=None):
    if checkpoint:
        run_config = checkpoint['run_config']
    else:
        run_config = {}
        run_config['dataset_id'] = 'dyna'

        # TODO(ff-k): The paper says nothing about batch size or stuff like batch normalization. Let's experiment it ourselves
        run_config['batch_size'] = 16

        run_config['lstm_num_layers'] = 3           # NOTE(ff-k): See Section 4.1
        run_config['lstm_hidden_dimensions'] = 128  # NOTE(ff-k): See Section 4.1
        run_config['n_frames'] = 32                 # NOTE(ff-k): See Section 4.1
        run_config['adam_learning_rate'] = 0.001    # NOTE(ff-k): Not specified in the paper
        run_config['adam_beta_1'] = 0.9             # NOTE(ff-k): See Section 4.1
        run_config['adam_beta_2'] = 0.999           # NOTE(ff-k): See Section 4.1
        run_config['adam_eps'] = 1e-8               # NOTE(ff-k): Not specified in the paper
        run_config['epoch'] = 0

        run_config['feature_vector_length'] = 9   # TODO(ff-k): Dataset dependent param, convert to argument
        run_config['vertex_count_of_mesh'] = 6890 # TODO(ff-k): Dataset dependent param, convert to argument
        
    return run_config

def main(mode, force_dataset, checkpoint_path):
    
    if checkpoint_path == '':
        checkpoints_path = '../checkpoints/'
        checkpoints = sorted([fp for fp in os.listdir(checkpoints_path) if fp.endswith('pt')])
        if len(checkpoints) > 0:
            checkpoint_path = checkpoints_path + checkpoints[len(checkpoints)-1]
            if mode == 'train':
                print('Make sure to clean checkpoints directory if you want to start training from scratch')
        else:
            checkpoint = {}
            if mode == 'test':
                print('Could not find a pretrained network. Terminating...')
                return
    else:
        checkpoint = torch.load(checkpoint_path)

    if checkpoint_path != '':
        print('Using checkpoint: %s' % (checkpoint_path))
        checkpoint = torch.load(checkpoint_path)

    cfg = get_run_config(checkpoint)
    if force_dataset != '':
        cfg['dataset_id'] = force_dataset

    print('Running | mode: %s, dataset: %s' % (mode, cfg['dataset_id']));

    pytorch_model = SMDSwBdLSTMNet(cfg['feature_vector_length'], cfg['vertex_count_of_mesh'], cfg['lstm_num_layers'], cfg['lstm_hidden_dimensions'], cfg['n_frames'])
    pytorch_l2_loss_fwd = nn.L1Loss() # TODO(ff-k): Multiply with n to cancel 'mean' as it actually computes MAE, not L1
    pytorch_l2_loss_bwd = nn.L1Loss() # TODO(ff-k): Multiply with n to cancel 'mean' as it actually computes MAE, not L1

    if checkpoint:
        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
        pytorch_l2_loss_fwd.load_state_dict(checkpoint['loss_fwd_fn_state_dict'])
        pytorch_l2_loss_bwd.load_state_dict(checkpoint['loss_bwd_fn_state_dict'])

    if mode == 'train':
        pytorch_dataset = SMDSwBdLSTMDataset('../data/' + cfg['dataset_id'] + '/train')
        pytorch_data_loader = data_utils.DataLoader(pytorch_dataset, batch_size=cfg['batch_size'], shuffle=True)
        pytorch_optimizer = optim.Adam(pytorch_model.parameters(), lr=cfg['adam_learning_rate'], 
                                       betas=(cfg['adam_beta_1'], cfg['adam_beta_2']), eps=cfg['adam_eps'])
        if checkpoint:
            pytorch_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        pytorch_model.train()

        checkpoint_save_interval = 1
        epoch = cfg['epoch']
        while globals()['interruped_by_user'] == False:
            time_epoch_begin = perf_counter()
            for tensor_self_feature_vectors, tensor_neighbour_feature_vectors in pytorch_data_loader:
                # TODO(ff-k): Fix the order of dimensions after introducing dummy data
                initial_h_fwd = torch.full((cfg['lstm_num_layers'], cfg['batch_size'], cfg['lstm_hidden_dimensions']), 0.1)
                initial_c_fwd = torch.full((cfg['lstm_num_layers'], cfg['batch_size'], cfg['lstm_hidden_dimensions']), 0.1)
                predictions_fwd = pytorch_model(tensor_self_feature_vectors, tensor_neighbour_feature_vectors, initial_h_fwd, initial_c_fwd, False)
                
                initial_h_bwd = torch.full((cfg['lstm_num_layers'], cfg['batch_size'], cfg['lstm_hidden_dimensions']), -0.1)
                initial_c_bwd = torch.full((cfg['lstm_num_layers'], cfg['batch_size'], cfg['lstm_hidden_dimensions']), -0.1)
                predictions_bwd = pytorch_model(tensor_self_feature_vectors, tensor_neighbour_feature_vectors, initial_h_bwd, initial_c_bwd, True)

                # TODO(ff-k): L_recon_fwd, L_recon_bwd, L_bidir, L_reg_2, L_KL
                # loss_fwd = pytorch_l2_loss_fwd(prediction, tensor_noisy)
                # loss_bwd = pytorch_l2_loss_bwd(prediction, tensor_noisy)
                
                pytorch_optimizer.zero_grad()
                # loss_fwd.backward()
                # loss_bwd.backward()
                pytorch_optimizer.step()
            
            time_epoch_end = perf_counter()
            if epoch % checkpoint_save_interval == 0 or globals()['interruped_by_user'] == True:
                cfg['epoch'] = epoch + 1
                torch.save({
                    'avg_epoch_loss': avg_epoch_loss,
                    'run_config': cfg,
                    'model_state_dict': pytorch_model.state_dict(),
                    'optimizer_state_dict': pytorch_optimizer.state_dict(),
                    'loss_fwd_fn_state_dict': pytorch_l2_loss_fwd.state_dict(),
                    'loss_bwd_fn_state_dict': pytorch_l2_loss_bwd.state_dict(),
                    }, checkpoints_path + 'checkpoint_%05d.pt' % (epoch))

            epoch = epoch + 1
    elif mode == 'test':
        pytorch_model.eval()

        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')

        # TODO(ff-k)

    else:
        raise NotImplementedError('Unknown running mode: \'%s\'' % mode)

if __name__ == "__main__":

    # NOTE(ff-k): The entry function and some basic stuff like handling checkpoints, signals etc. are based 
    #             on my previous implementation for the last year's CENG501 project:
    #             https://github.com/sinankalkan/CENG501-Spring2021/blob/main/project_Kucukdemir/code/main.py

    mode = 'test'
    force_dataset = ''
    checkpoint_path = ''

    argc = len(sys.argv)
    if argc > 3:
        mode = sys.argv[1]
        force_dataset = sys.argv[2]
        checkpoint_path = sys.argv[3]
    elif argc > 2:
        mode = sys.argv[1]
        force_dataset = sys.argv[2]
    elif argc > 1:
        mode = sys.argv[1]
    
    if mode == 'train':
        globals()['interruped_by_user'] = False
        signal.signal(signal.SIGINT, interrupt_handler)
    
    # NOTE(ff-k): set random_seed to 0 or some other constant value to repeat results
    random_seed = int(time.time()*1000)%(2**32-1) 
    np.random.seed(random_seed)
    
    main(mode, force_dataset, checkpoint_path)