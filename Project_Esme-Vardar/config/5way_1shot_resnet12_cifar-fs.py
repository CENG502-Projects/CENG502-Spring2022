from collections import OrderedDict

config = OrderedDict()
config['dataset_name'] = 'cifar-fs'
config['backbone'] = 'resnet12'
config['emb_size'] = 128
config['num_generation'] = 6
config['num_loss_generation'] = 6
config['generation_weight'] = 0.2
config['point_distance_metric'] = 'l2'
config['distribution_distance_metric'] = 'l2'

train_opt = OrderedDict()
train_opt['num_ways'] = 5
train_opt['num_shots'] = 1
train_opt['batch_size'] = 20
train_opt['iteration'] = 100000
train_opt['lr'] = 1e-3
train_opt['weight_decay'] = 1e-3
train_opt['dec_lr'] = 15000
train_opt['dropout'] = 0.1
train_opt['lr_adj_base'] = 0.1
train_opt['loss_indicator'] = [1, 1, 0]
train_opt["num_queries"] = 1

eval_opt = OrderedDict()
eval_opt["num_queries"] = 1
eval_opt['num_ways'] = 5
eval_opt['num_shots'] = 1
eval_opt['batch_size'] = 20
eval_opt['iteration'] = 250
eval_opt['interval'] = 500
config['train_config'] = train_opt
config['eval_config'] = eval_opt
