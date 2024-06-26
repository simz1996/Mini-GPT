import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_heads: int = 32
    n_kv_heads:Optional[int] = None
    vocab_size: int = -1
    multiple_of: int =256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None



def precompute_theta_pos_frequencies(head_dim:int,seq_len:int,device:str,theta:float =10000):
    assert head_dim % 2 == 0, "Dimension must be divisable by two"

    theta_numerator = torch.arange(0, head_dim, 2).float()

    theta = 1.0 / (theta **(theta_numerator/ head_dim)).to(device)

    m = torch.arange(seq_len, device=device)

    freq = torch.outer(m, theta).float()

    freq_complex= torch.polar(torch.ones_like(fre q),freq)

    return freq_complex

def apply_rotary_embeddings(x: torch.Tensor, freq_complex: torch.Tensor, device:str):
    x_complex = torch.view_as_complex(x.float(x.float().reshape(*x.shape[:-1], -1, 2)))
    freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freq_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size,seq_len,n_kv_heads* n_rep,head_dim
    if n_rep ==1
      return x

    else:
        return (x[:,:,:, None, :]).expand(batch_size,seq_len,n_kv_heads,n_reps,head_dim).reshape((batch_size, seq_len, n_kv_heads * n_rep, head_dim))
                

    
class RMSNorm(nn.Module):

    def __init__(self, dim:int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) 

    def norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True + self.eps))
    
    def forward(self, x:torch.Tensor):
        return self.weight + self._norm(x.float()).type_as(x)
    

class SelfAttentionBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias = False)  
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)   
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = Fals)   
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias = False)

        self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.n_heads_q) 
        args.cache_v = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)

    def forward(self, x:torch.Tensor, start_pos:int, freq_complex: torch.Tensor): 
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        xv = xv.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)

        xq = apply_rotary_embeddings(xq, freq_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freq_complex, device=x.device)  
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xv

        keys = self.cache_k[:batch_size, 0:start_pos + seq_len]
        values = self.cache_v[:batch, 0:start_pos + seq_len]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values,self.n_rep)

        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim = -1).type_as(xq)
        output = torch.matmul(scores, values)

        output = (output.transpose(1,2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output)
    

class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2* hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

            hidden_dim =  args.multiple_of* ((hidden + args.multiple_of -1)// args.multiple_of)

            self.w1 = nn.Linear(args.dim, hidden_dim, bias = False)
            self.w2 = nn.Linear(args.dim, hidden_dim, bias = False)
            self.w3 = nn.Linear(args.dim, hidden_dim, bias = False)   

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x= self.w2(x)
        return x
           
           
        

    

class EncoderBlock(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim//args.n_heads

        self.attention = SelfAttentionBlock(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps = args.norm_eps)
        self.ffwrd_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x:torch.Tensor, start_pos:int, freq_complex:torch.Tensor):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freq_complex)
        out = h + self.feed_forward.forward(self.ffwrd_norm(h))
        




class Transformer(nn.module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"
        
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_layers = args.n_layer
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        #self.n_heads = args.n_heads
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

            self.norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.out = nn.Linear(args.dim, self.vocab_size, bias= False)

            self.freq_complex = precompute_theta_pos_frequencies(self.args.dim//self.args.n_heads, self.args.max_seq_len*2,device = self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size,seq_len = tokens.shape

        assert seq_len == 1, "only one token at a time can be processed"

        h=self.tok_embeddings(tokens)

        freqs_complex = self.freq_complex[start_pos:start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, self.freq_complex)

        h=self.norm(h)
        output = self.output(h).float()
        return output







        