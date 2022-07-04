from backbone import ResNet12
from utils import set_logging_config, adjust_learning_rate, save_checkpoint, allocate_tensors, preprocessing, \
    initialize_nodes_edges, backbone_two_stage_initialization, one_hot_encode
from dataloader import MiniImagenet, TieredImagenet, Cifar, CUB200, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random
import logging
import argparse
import imp
from mutual_crf_gnn import MCRFGNN
from crf import binary_comp_calc
from crf import CRF 
from gnn_only import GNN_module


class MCRFGNNTrainer(object):
    def __init__(self, enc_module, gnn_module, data_loader, log, arg, config, best_step):

        self.arg = arg
        self.config = config
        self.train_opt = config['train_config']
        self.eval_opt = config['eval_config']

        # initialize variables
        self.tensors = allocate_tensors()
        for key, tensor in self.tensors.items():
            self.tensors[key] = tensor.to(self.arg.device)

        # set backbone and DPGN
        self.enc_module = enc_module.to(arg.device)
        self.gnn_module = gnn_module.to(arg.device)

        self.ce_loss = nn.CrossEntropyLoss()
        # set logger
        self.log = log

        # get data loader
        self.data_loader = data_loader

        # set parameters
        self.module_params = list(self.enc_module.parameters()) + list(self.gnn_module.parameters())

        # set optimizer
        self.optimizer = optim.Adam(
            params=self.module_params,
            lr=self.train_opt['lr'],
            weight_decay=self.train_opt['weight_decay'])

        # set loss
        self.edge_loss = nn.BCELoss(reduction='none')
        self.pred_loss = nn.CrossEntropyLoss(reduction='none')

        # initialize other global variables
        self.global_step = best_step
        self.best_step = best_step
        self.val_acc = 0
        self.test_acc = 0

    def train(self):
        """
        train function
        :return: None
        """

        num_supports, num_samples, query_edge_mask, evaluation_mask = \
            preprocessing(self.train_opt['num_ways'],
                          self.train_opt['num_shots'],
                          self.train_opt['num_queries'],
                          self.train_opt['batch_size'],
                          self.arg.device)


        # main training loop, batch size is the number of tasks
        for iteration, batch in enumerate(self.data_loader['train']()):
            # init grad
            self.optimizer.zero_grad()

            # set current step
            self.global_step += 1

            # initialize nodes and edges for dual graph model
            all_data, all_label_in_edge, unary_comp, support_label, \
                 query_label, all_label = initialize_nodes_edges(batch,
                                                                      num_supports,
                                                                      self.tensors,
                                                                      self.train_opt['batch_size'],
                                                                      self.train_opt['num_queries'],
                                                                      self.train_opt['num_ways'],
                                                                      self.arg.device)

            # set as train mode
            self.enc_module.train()
            self.gnn_module.train()
            
            last_layer_data_temp = []
            all_data = all_data[:]
            for data in all_data.chunk(all_data.size(1), dim=1):
                encoded_result = self.enc_module(data.squeeze(1))
                last_layer_data_temp.append(encoded_result[0])
            enc_feat = torch.stack(last_layer_data_temp, dim=1)

            if self.arg.arch == "mcrfgnn":
                binary_comp = binary_comp_calc(enc_feat, all_label, \
                    self.train_opt['batch_size'], self.arg.device,self.train_opt['num_ways'])

                
                #elf.gnn_module.init_crf(enc_feat, self.train_opt['batch_size'],enc_feat.size()[1],self.train_opt['num_ways'], \
                #   unary_comp, binary_comp, num_supports, all_label_in_edge)
                

                # run the DPGN model
                aff_mat, belief_list, aff_list = self.gnn_module(enc_feat, \
                    self.train_opt['batch_size'],enc_feat.size()[1],self.train_opt['num_ways'], \
                    unary_comp, binary_comp, num_supports, all_label_in_edge, all_data, all_label)
                

                crf_loss = self.compute_crf_loss(num_supports, all_label, belief_list)
                gnn_loss = self.compute_gnn_loss(num_supports, aff_mat, all_label,aff_list)
                lamd_crf = 0.1
                lamd_gnn = 1

                total_loss = torch.autograd.Variable(crf_loss * lamd_crf + gnn_loss * lamd_gnn,requires_grad = True)

                total_loss.backward()

                self.optimizer.step()

                # adjust learning rate
                adjust_learning_rate(optimizers=[self.optimizer],
                                    lr=self.train_opt['lr'],
                                    iteration=self.global_step,
                                    dec_lr_step=self.train_opt['dec_lr'],
                                    lr_adj_base =self.train_opt['lr_adj_base'])



                if self.global_step % self.eval_opt['interval'] == 0:
                    is_best = 0
                    test_acc = self.eval(partition='test')
                    if test_acc > self.test_acc:
                        is_best = 1
                        self.test_acc = test_acc
                        self.best_step = self.global_step

                    # log evaluation info
                    self.log.info('test_acc : {}         step : {} '.format(test_acc, self.global_step))
                    self.log.info('test_best_acc : {}    step : {}'.format( self.test_acc, self.best_step))

                    # save checkpoints (best and newest)
                    save_checkpoint({
                        'iteration': self.global_step,
                        'enc_module_state_dict': self.enc_module.state_dict(),
                        'gnn_module_state_dict': self.gnn_module.state_dict(),
                        'test_acc': self.test_acc,
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best, os.path.join(self.arg.checkpoint_dir, self.arg.exp_name))
                
            else:
                ind = np.arange(5,10,1)
                np.random.shuffle(ind)
                query_feat = enc_feat[:,ind,:]
                tmpfeat = torch.cat((enc_feat[:,0:5,:],query_feat[:,0,:].unsqueeze(1)),dim=1)

                support_one_hot = torch.nn.functional.one_hot(support_label,self.train_opt['num_ways'])
                
                logsoft_prob = self.gnn_module(tmpfeat, support_one_hot).squeeze()
                loss = F.nll_loss(logsoft_prob, query_label[:,ind-5][:,0])
                loss.backward()
                self.optimizer.step()

                adjust_learning_rate(optimizers=[self.optimizer],
                    lr=self.train_opt['lr'],
                    iteration=self.global_step,
                    dec_lr_step=self.train_opt['dec_lr'],
                    lr_adj_base =self.train_opt['lr_adj_base'])



                if self.global_step % self.eval_opt['interval'] == 0:
                    is_best = 0
                    test_acc = self.eval(partition='test')
                    if test_acc > self.test_acc:
                        is_best = 1
                        self.test_acc = test_acc
                        self.best_step = self.global_step

                    # log evaluation info
                    self.log.info('test_acc : {}         step : {} '.format(test_acc, self.global_step))
                    self.log.info('test_best_acc : {}    step : {}'.format( self.test_acc, self.best_step))

                    # save checkpoints (best and newest)
                    save_checkpoint({
                        'iteration': self.global_step,
                        'enc_module_state_dict': self.enc_module.state_dict(),
                        'gnn_module_state_dict': self.gnn_module.state_dict(),
                        'test_acc': self.test_acc,
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best, os.path.join(self.arg.checkpoint_dir, self.arg.exp_name))

    def eval(self, partition='test', log_flag=True):
        """
        evaluation function
        :param partition: which part of data is used
        :param log_flag: if log the evaluation info
        :return: None
        """

        num_supports, num_samples, query_edge_mask, evaluation_mask = preprocessing(
            self.eval_opt['num_ways'],
            self.eval_opt['num_shots'],
            self.eval_opt['num_queries'],
            self.eval_opt['batch_size'],
            self.arg.device)

        loss_list = []
        # main training loop, batch size is the number of tasks
        for current_iteration, batch in enumerate(self.data_loader[partition]()):

            # initialize nodes and edges for dual graph model
            all_data, all_label_in_edge, unary_comp, support_label, \
                 query_label, all_label = initialize_nodes_edges(batch,
                                                                      num_supports,
                                                                      self.tensors,
                                                                      self.eval_opt['batch_size'],
                                                                      self.eval_opt['num_queries'],
                                                                      self.eval_opt['num_ways'],
                                                                      self.arg.device)

            # set as eval mode
            self.enc_module.eval()
            self.gnn_module.eval()

            last_layer_data_temp = []
            for data in all_data.chunk(all_data.size(1), dim=1):
                encoded_result = self.enc_module(data.squeeze(1))
                last_layer_data_temp.append(encoded_result[0])
            enc_feat = torch.stack(last_layer_data_temp, dim=1)
            if self.arg.arch == "mcrfgnn":
                binary_comp = binary_comp_calc(enc_feat, all_label, \
                    self.eval_opt['batch_size'], self.arg.device, self.eval_opt['num_ways'])

            #elf.gnn_module.init_crf(enc_feat, self.train_opt['batch_size'],enc_feat.size()[1],self.train_opt['num_ways'], \
            #   unary_comp, binary_comp, num_supports, all_label_in_edge)

            
                # run the DPGN model
                aff_mat, belief_list, aff_list = self.gnn_module(enc_feat, \
                    self.eval_opt['batch_size'],enc_feat.size()[1],self.eval_opt['num_ways'], \
                    unary_comp, binary_comp, num_supports, all_label_in_edge, all_data, all_label)

                preds = torch.argmax(belief_list[len(belief_list)-1][:,:,:,6],dim=2)
                item_mean = torch.mean(torch.eq(preds[:,num_supports:], all_label[:,num_supports:]).float())
                loss_list.append(item_mean.item())
                
            else:
                ind = np.arange(5,10,1)
                np.random.shuffle(ind)
                query_feat = enc_feat[:,ind,:]
                tmpfeat = torch.cat((enc_feat[:,0:5,:],query_feat[:,0,:].unsqueeze(1)),dim=1)

                support_one_hot = torch.nn.functional.one_hot(support_label,self.train_opt['num_ways'])
                
                logsoft_prob = self.gnn_module(tmpfeat, support_one_hot)
                pred = torch.argmax(logsoft_prob, dim=1)

                item_mean = torch.mean(torch.eq(pred, query_label[:,ind-5][:,0]).float())
                loss_list.append(item_mean.item())


        # logging
        if log_flag:
            self.log.info('------------------------------------')
            self.log.info('evaluation: total_count=%d, accuracy: mean=%.2f%%' %
                          (current_iteration,
                           np.array(loss_list).mean() * 100,
                           ))
            self.log.info('------------------------------------')

        return np.array(loss_list).mean()

    def compute_crf_loss(self,num_supports, all_label, belief_list):
        #Loss larda weight meani olmayacak 0.2 parametre şeklinde weight detay bakılmadı
        crf_loss = 0
        for i in range(num_supports, all_label.size()[1]):
            for j in range(0,len(belief_list)):
                #crf_loss += crf_meanw_list[j] * self.ce_loss(belief_list[j][:,i,:], all_label[:,i])
                if j == (len(belief_list) - 1):
                    crf_loss += 1 * self.ce_loss(belief_list[j][:,i,:,6], all_label[:,i])
                else:
                    crf_loss += 0.2 * self.ce_loss(belief_list[j][:,i,:,6], all_label[:,i])
        return crf_loss 
                

    def compute_gnn_loss(self,num_supports, aff_mat, all_label,aff_list):
        gnn_loss = 0
        for i in range(num_supports, len(all_label[1])):
            for j in range(num_supports):
                for k in range(len(aff_list)):
                    label_eq_tens = torch.eq(all_label[:,i], all_label[:,j])
                    edge_label_mat = torch.zeros(aff_list[k][:,i,j].size()).to(self.arg.device)
                    edge_label_mat[label_eq_tens] = 1
                    if k == (len(aff_list) - 1):
                        gnn_loss += 1 * F.binary_cross_entropy(aff_list[k][:,i,j],edge_label_mat)
                    else:
                        gnn_loss += 0.2 * F.binary_cross_entropy(aff_list[k][:,i,j],edge_label_mat)
                    
        return gnn_loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='gpu device number of using')

    parser.add_argument('--config', type=str, default=os.path.join('.', 'config', '5way_1shot_resnet12_mini-imagenet.py'),
                        help='config file with parameters of the experiment. '
                             'It is assumed that the config file is placed under the directory ./config')

    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join('.', 'checkpoints'),
                        help='path that checkpoint will be saved and loaded. '
                             'It is assumed that the checkpoint file is placed under the directory ./checkpoints')

    parser.add_argument('--num_gpu', type=int, default=1,
                        help='number of gpu')

    parser.add_argument('--arch', type=str, default="mcrfgnn",
                        help='number of gpu')

    parser.add_argument('--display_step', type=int, default=100,
                        help='display training information in how many step')

    parser.add_argument('--log_step', type=int, default=100,
                        help='log information in how many steps')

    parser.add_argument('--log_dir', type=str, default=os.path.join('.', 'logs'),
                        help='path that log will be saved. '
                             'It is assumed that the checkpoint file is placed under the directory ./logs')

    parser.add_argument('--dataset_root', type=str, default='./data',
                        help='root directory of dataset')

    parser.add_argument('--seed', type=int, default=222,
                        help='random seed')

    parser.add_argument('--mode', type=str, default='train',
                        help='train or eval')

    args_opt = parser.parse_args()

    config_file = args_opt.config

    # Set train and test datasets and the corresponding data loaders
    config = imp.load_source("", config_file).config
    train_opt = config['train_config']
    eval_opt = config['eval_config']

    args_opt.exp_name = '{}way_{}shot_{}_{}'.format(train_opt['num_ways'],
                                                    train_opt['num_shots'],
                                                    config['backbone'],
                                                    config['dataset_name'])
    train_opt['num_queries'] = 1
    eval_opt['num_queries'] = 1
    set_logging_config(os.path.join(args_opt.log_dir, args_opt.exp_name))
    logger = logging.getLogger('main')

    # Load the configuration params of the experiment
    logger.info('Launching experiment from: {}'.format(config_file))
    logger.info('Generated logs will be saved to: {}'.format(args_opt.log_dir))
    logger.info('Generated checkpoints will be saved to: {}'.format(args_opt.checkpoint_dir))
    print()

    logger.info('-------------command line arguments-------------')
    logger.info(args_opt)
    print()
    logger.info('-------------configs-------------')
    logger.info(config)

    # set random seed
    np.random.seed(args_opt.seed)
    torch.manual_seed(args_opt.seed)
    torch.cuda.manual_seed_all(args_opt.seed)
    random.seed(args_opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if config['dataset_name'] == 'mini-imagenet':
        dataset = MiniImagenet
        print('Dataset: MiniImagenet')
    elif config['dataset_name'] == 'tiered-imagenet':
        dataset = TieredImagenet
        print('Dataset: TieredImagenet')
    elif config['dataset_name'] == 'cifar-fs':
        dataset = Cifar
        print('Dataset: Cifar')
    elif config['dataset_name'] == 'cub-200-2011':
        dataset = CUB200
        print('Dataset: CUB200')
    else:
        logger.info('Invalid dataset: {}, please specify a dataset from '
                    'mini-imagenet, tiered-imagenet, cifar-fs and cub-200-2011.'.format(config['dataset_name']))
        exit()

    cifar_flag = True if args_opt.exp_name.__contains__('cifar') else False
    if config['backbone'] == 'resnet12':
        enc_module = ResNet12(emb_size=config['emb_size'], cifar_flag=cifar_flag)
        print('Backbone: ResNet12')
    else:
        logger.info('Invalid backbone: {}, please specify a backbone model from '
                    'convnet or resnet12.'.format(config['backbone']))
        exit()


    print(args_opt.arch)
    if args_opt.arch == "mcrfgnn":
        gnn_module = MCRFGNN(train_opt['num_ways'], args_opt.device)
    elif args_opt.arch == "gnn":
        gnn_module = GNN_module(train_opt['num_ways'], 133, 64, 3, feature_type='dense')
    # multi-gpu configuration
    [print('GPU: {}  Spec: {}'.format(i, torch.cuda.get_device_name(i))) for i in range(args_opt.num_gpu)]

    if args_opt.num_gpu > 1:
        print('Construct multi-gpu model ...')
        enc_module = nn.DataParallel(enc_module, device_ids=range(args_opt.num_gpu), dim=0)
        gnn_module = nn.DataParallel(gnn_module, device_ids=range(args_opt.num_gpu), dim=0)
        print('done!\n')

    if not os.path.exists(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)):
        os.makedirs(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name))
        logger.info('no checkpoint for model: {}, make a new one at {}'.format(
            args_opt.exp_name,
            os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)))
        best_step = 0
    else:
        if not os.path.exists(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name, 'model_best.pth.tar')):
            best_step = 0
        else:
            logger.info('find a checkpoint, loading checkpoint from {}'.format(
                os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)))
            best_checkpoint = torch.load(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name, 'model_best.pth.tar'))

            logger.info('best model pack loaded')
            best_step = best_checkpoint['iteration']
            enc_module.load_state_dict(best_checkpoint['enc_module_state_dict'])
            gnn_module.load_state_dict(best_checkpoint['gnn_module_state_dict'])
            logger.info('current best test accuracy is: {}, at step: {}'.format(best_checkpoint['test_acc'], best_step))

    dataset_train = dataset(root=args_opt.dataset_root, partition='train')
    dataset_valid = dataset(root=args_opt.dataset_root, partition='val')
    dataset_test = dataset(root=args_opt.dataset_root, partition='test')

    train_loader = DataLoader(dataset_train,
                              num_tasks=train_opt['batch_size'],
                              num_ways=train_opt['num_ways'],
                              num_shots=train_opt['num_shots'],
                              num_queries=train_opt['num_queries'],
                              epoch_size=train_opt['iteration'])
    valid_loader = DataLoader(dataset_valid,
                              num_tasks=eval_opt['batch_size'],
                              num_ways=eval_opt['num_ways'],
                              num_shots=eval_opt['num_shots'],
                              num_queries=eval_opt['num_queries'],
                              epoch_size=eval_opt['iteration'])
    test_loader = DataLoader(dataset_test,
                             num_tasks=eval_opt['batch_size'],
                             num_ways=eval_opt['num_ways'],
                             num_shots=eval_opt['num_shots'],
                             num_queries=eval_opt['num_queries'],
                             epoch_size=eval_opt['iteration'])

    data_loader = {'train': train_loader,
                   'val': valid_loader,
                   'test': test_loader}

    # create trainer
    trainer = MCRFGNNTrainer(enc_module=enc_module,
                           gnn_module=gnn_module,
                           data_loader=data_loader,
                           log=logger,
                           arg=args_opt,
                           config=config,
                           best_step=best_step)

    if args_opt.mode == 'train':
        trainer.train()
    elif args_opt.mode == 'eval':
        trainer.eval()
    else:
        print('select a mode')
        exit()


if __name__ == '__main__':
    main()
