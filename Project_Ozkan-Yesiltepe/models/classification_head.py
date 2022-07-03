import torch
import torch.nn as nn
from interpreter import NeuralInterpreter


class NeuralInterpreter_vision(nn.Module):
  '''

  Args:
  -----
    ns            [int]:       Number of `scripts`
    ni            [int]:       Number of `function iterations` in a script
    nf            [int]:       Number of `functions` per iteration
    din           [int]:       Dimension of the input  projection
    dcond         [int]:       Dimension of the code vector
    mlp_depth     [int]:       Number of MLP depths of LOC layer
    n_heads       [int]:       Number of attention heads
    signature_dim [int]:       Dimension of `signature` vector
    treshold      [int]:       Treshold value for routing
    code_dim      [int]:       Dimension of `code` vector
    n_classes     [int]:       Number of classification heads
    attn_prob     [float]:     Drop-out rate
    proj_prob     [float]:     Drop-out rate
  '''
  def __init__( self, 
                ns, 
                ni, 
                nf, 
                din, 
                dcond, 
                mlp_depth, 
                nheads,
                type_inference_width, 
                signature_dim, 
                threshold, 
                code_dim, 
                n_classes=10,
                attn_prob=0, 
                proj_prob=0
              ) 

    super().__init__()
    self.ni_model = NeuralInterpreter( ns, 
                                       ni, 
                                       nf, 
                                       din, 
                                       dcond, 
                                       mlp_depth, 
                                       nheads,
                                       type_inference_width, 
                                       signature_dim, 
                                       threshold,  
                                       code_dim, 
                                       attn_prob=0, 
                                       proj_prob=0)

    self.cls_head = nn.Linear(din, n_classes)
  
  def forward(self, x):
    x = self.ni_model(x)
    # first cls taken
    x = x[:,0,:] 
    x = self.cls_head(x)
    return x
