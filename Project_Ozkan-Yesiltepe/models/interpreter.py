import torch
import torch.nn as nn

from basic layers import PatchEmbedding, MLP, TypeMatching, ModLin2D, ModMLP, ModAttn  

class LOC(nn.Module):
  '''
  Line of Code Layer Composed of 1 attention + 1 MLP layers
  Args:
  -----
    code_matrix [Tensor()]:  Code matrix of a all `function`s.
    din         [int]:       Dimension of the input  projection
    dcond       [int]:       Dimension of the code vector
    n_heads     [int]:       Number of attention heads
    mlp_depth   [int]:       Number of MLP depths of LOC layer
    typematch   [nn.Module]: TypeMatching Module
    w_c         [Tensor()]:  Projection Matrix of `code` vector
    W           [Tensor()]:  Projection Matrix of Fusion operation 
    b           [Tensor()]:  Bias vector of Fusion operation
    W_qkv       [Tensor()]:  Weight matrix 
    b_qkv       [Tensor()]:  Bias vector
    attn_prob   [float]:     Drop-out rate
    proj_prob   [float]:     Drop-out rate
  '''
  def __init__( self, 
                code_matrix, 
                din, 
                dcond, 
                n_heads, 
                mlp_depth, 
                typematch, 
                w_c, 
                W, 
                b, 
                W_qkv, 
                b_qkv,
                attn_prob=0, 
                proj_prob=0):

    super().__init__()

    self.typematch = typematch
    self.norm1 = torch.nn.LayerNorm(din)
    self.norm2 = torch.nn.LayerNorm(din)

    self.modattn = ModAttn( code_matrix, 
                            din, 
                            dcond, 
                            n_heads, 
                            w_c, 
                            W, 
                            b, 
                            W_qkv, 
                            b_qkv,
                            attn_prob, 
                            proj_prob)

    self.modmlp = ModMLP( mlp_depth, 
                          code_matrix, 
                          din, 
                          din, 
                          dcond,
                          w_c, 
                          W, 
                          b)

  def forward(self, x):
    compat_matrix = self.typematch(x)
    # x = x.squeeze()
    x_norm = self.norm1(x)
    a_hat = self.modattn(x_norm, compat_matrix)
    
    compat_matrix = compat_matrix.unsqueeze(-1)
    a = x.unsqueeze(1) + compat_matrix*a_hat
    
    b_hat = self.modmlp(self.norm2(a))
    y = a + compat_matrix*b_hat
    
    # pool-LOC => eqn-11
    y = x + torch.sum(compat_matrix*y, dim=1)
    return y


class Script(nn.Module):
  '''
  Script blocks composed of LOC blocks
  
  Assumption:
  -----------
    LOC is composed of 1 layer.
  
  Args:
  -----
    ni            [int]:       Number of function iterations in a script
    nf            [int]:       Number of functions per iteration
    din           [int]:       Dimension of the input  projection
    dcond         [int]:       Dimension of the code vector
    n_heads       [int]:       Number of attention heads
    mlp_depth     [int]:       Number of MLP depths of LOC layer
    typematch     [nn.Module]: TypeMatching Module
    treshold      [int]:       Treshold value for routing 
    code_dim      [int]:       Dimension of `code` vector
    signature_dim [int]:       Dimension of `signature` vector
    W             [Tensor()]:  Projection Matrix of Fusion operation 
    b             [Tensor()]:  Bias vector of Fusion operation
    W_qkv         [Tensor()]:  Weight matrix 
    b_qkv         [Tensor()]:  Bias vector
    attn_prob     [float]:     Drop-out rate
    proj_prob     [float]:     Drop-out rate
  '''
  
  def __init__( self, 
                ni, 
                nf, 
                din, 
                dcond, 
                n_heads, 
                mlp_depth, 
                type_inference, 
                threshold, 
                code_dim, 
                signature_dim,
                W, 
                b, 
                W_qkv, 
                b_qkv,
                attn_prob=0, 
                proj_prob=0):

    super().__init__()
    
    # w_c shared among all functions in a script  
    self.register_parameter('w_c', nn.Parameter(torch.empty(din, dcond)))
    nn.init.xavier_normal_(self.w_c)

    # high-entropy & fixed function signature => avoid mode collapse
    self.register_parameter('funcsign_matrix', nn.Parameter(torch.ones(nf, signature_dim)))
    nn.init.xavier_normal_(self.funcsign_matrix)
    
    self.register_parameter('code_matrix', nn.Parameter(torch.empty(code_dim, nf)))
    nn.init.xavier_normal_(self.code_matrix)

    self.typematch = TypeMatching(type_inference, self.funcsign_matrix, threshold)

    self.locBlocks = []
    for i in range(ni):
      # add LOC layer
      self.loc_blocks.append(LOC(self.
                                code_matrix, 
                                din, 
                                dcond, 
                                n_heads, 
                                mlp_depth, 
                                self.typematch, 
                                self.w_c, 
                                W, 
                                b, 
                                W_qkv, 
                                b_qkv, 
                                attn_prob, 
                                proj_prob))
      
    self.loc_blocks = nn.Sequential(*self.loc_blocks)
  
  def forward(self, x):
    x = self.loc_blocks(x)
    return x
    

class NeuralInterpreter(nn.Module):
  '''
  Neural Interpreter layer that operates the logic
  Args:
  -----
    ns            [int]:       Number of `scripts`
    ni            [int]:       Number of `function iterations` in a script
    nf            [int]:       Number of `functions` per iteration
    din           [int]:       Dimension of the input  projection
    dcond         [int]:       Dimension of the code vector
    n_heads       [int]:       Number of attention heads
    mlp_depth     [int]:       Number of MLP depths of LOC layer
    treshold      [int]:       Treshold value for routing 
    code_dim      [int]:       Dimension of `code` vector
    signature_dim [int]:       Dimension of `signature` vector
    treshold      [int]:       Treshold value for routing
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
                attn_prob=0, 
                proj_prob=0, 
              ):

    super().__init__()
    
    self.register_parameter('W', nn.Parameter(torch.empty(din, din)))
    nn.init.xavier_normal_(self.W)
    
    self.register_parameter('b', nn.Parameter(torch.empty(din)))
    nn.init.normal_(self.b)
    
    self.register_parameter('W_qkv', nn.Parameter(torch.empty(3*din, din)))
    nn.init.xavier_normal_(self.W_qkv)
    
    self.register_parameter('b_qkv', nn.Parameter(torch.empty(3*din)))
    nn.init.normal_(self.b_qkv)

    # type inference 
    self.type_inference = MLP(din, type_inference_width, signature_dim)

    self.script_blocks = []
    for i in range(ns):
      # Create Script Blocks
      self.script_blocks.append(Script( ni, 
                                        nf, 
                                        din, 
                                        dcond, 
                                        nheads, 
                                        mlp_depth, 
                                        self.type_inference, 
                                        threshold, 
                                        code_dim, 
                                        signature_dim,
                                        self.W, 
                                        self.b, 
                                        self.W_qkv, 
                                        self.b_qkv,
                                        attn_prob,
                                        proj_prob ))

    self.script_blocks = nn.Sequential(*self.script_blocks)

  def forward(self, x):
    return self.script_blocks(x)
