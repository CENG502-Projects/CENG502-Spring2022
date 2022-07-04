import argparse
import yaml
import os
import copy
import sys


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr_drop', default=400, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # * Backbone
    parser.add_argument('--backbone', default='lie_t2t', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * LIE
    parser.add_argument('--N', default=2, type=int,
                        help="layers of encoders with LIE")
    parser.add_argument('--K', default=8, type=int,
                        help="layers of encoders without LIE")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='../coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--cfg', default=None, help='Config file for experiment')
    return parser


def get_arguments():
    # override order goes as follow: defaults < config < command line
    parser = get_args_parser()
    # get default arguments from config.py
    args = parser.parse_args([])
    print("args", args)
    # get arguments from command line
    args_command, _ = parser._parse_known_args(sys.argv[1:], argparse.Namespace())
    print("args_command",args_command)
    # get the arguments given in the command line, and keep them
    difference = {}
    for key, value in vars(args_command).items():
        difference[key] = value

    # update args wrt config file if given in the command line
    try:
         arg_cmd = args_command.cfg
    except AttributeError:
        arg_cmd = None
            
    if arg_cmd is not None:
        with open(arg_cmd) as file:
            config = yaml.safe_load(file)
        for key, value in config.items():
            if args.__contains__(key):
                args.__setattr__(key, value)
            else:
                print("key {} not recognized".format(key))
                raise ValueError
    # finally, update args from the difference
    for key, value in difference.items():
        args.__setattr__(key, value)

    return args


def store_yaml(args):
    config_path = os.path.join(args.output_dir, 'config.yaml')
    args_dict = copy.copy(vars(args))
    args_dict.pop('cfg', None)
    args_dict.pop('resume', None)
    args_dict.pop('output_dir', None)
    with open(config_path, 'w+') as file:
        yaml.dump(args_dict, file, default_flow_style=False)


