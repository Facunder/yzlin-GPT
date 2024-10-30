import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import inspect

# Model parameters
from dataclasses import dataclass
@dataclass
class ModelArgs:
    block_size: int = 1024  # Maximum input size
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0  # Default no dropout
    bias: bool = True  # True: includes bias in Linears and LayerNorms, like GPT-2. False: slightly better and faster

class RMSNorm(nn.Module):
    # RMS Norm Layer as used in llama
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        # Introducing eps to avoid division by zero

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        sqrt_mean_sq = torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True))
        # Computes the RMS Norm as defined
        return self.weight * hidden_states / (sqrt_mean_sq + self.eps)

class FlashAttention(nn.Module):
    # Flash attention as seen in NanoGPT
    def __init__(self, args):
        super().__init__()
        # Merging Q, K, V into one linear layer
        self.qkv_attention = nn.Linear(args.n_embed, 3 * args.n_embed, bias=args.bias)
        # Head size should equal sequence length as per a research paper
        self.n_head = args.n_head
        self.n_embed = args.n_embed
        # Compute head size
        assert args.n_embed % args.n_head == 0
        self.head_size = args.n_embed // args.n_head
        # Dropout
        self.dropout = args.dropout
        self.attention_dropout = nn.Dropout(self.dropout)
        # Projection layer
        self.context_projection = nn.Linear(self.n_embed, args.n_embed, bias=args.bias)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv_attention(x).split(self.n_embed, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Use torch's built-in flash attention
        y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                       dropout_p=self.dropout if self.training else 0,
                                                       is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Apply dropout to the output of the projection layer
        return self.attention_dropout(self.context_projection(y))

class MLP(nn.Module):
    # MLP structure as seen in llama
    def __init__(self, args):
        super().__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.up_projection = nn.Linear(args.n_embed, 4 * args.n_embed, bias=args.bias)
        self.down_projection = nn.Linear(4 * args.n_embed, args.n_embed, bias=args.bias)
        # Use relu activation function
        self.activation_function = nn.functional.relu
        # Introduce a gating mechanism as in llama
        self.gate = nn.Linear(args.n_embed, 4 * args.n_embed, bias=args.bias)

    def forward(self, x):
        gate_projection = self.gate(x)
        x = self.up_projection(x)
        x = self.activation_function(gate_projection) * x  # Element-wise multiplication with gate
        x = self.down_projection(x)
        return self.dropout(x)

class Block(nn.Module):
    # Block to stack later
    def __init__(self, args):
        super().__init__()
        self.norm = RMSNorm(args.n_embed)
        self.attention = FlashAttention(args)
        self.mlp = MLP(args)

    def forward(self, x):
        # Use pre-normalization
        x = x + self.attention(self.norm(x))  # Residual connection
        return x + self.mlp(self.norm(x))  # Residual link

class yzlinGPT(nn.Module):
    # Hybrid of llama and GPT-2
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            token_embedding = nn.Embedding(args.vocab_size, args.n_embed),
            position_embedding = nn.Embedding(args.block_size, args.n_embed),
            dropout = nn.Dropout(args.dropout),
            blocks = nn.ModuleList([Block(args) for _ in range(args.n_layer)]),
            norm = RMSNorm(args.n_embed)
        ))

        self.lm_head = nn.Linear(args.n_embed, args.vocab_size, bias=False)
        self.transformer.token_embedding.weight = self.lm_head.weight
        # Share weights between token_embedding and lm_head

        self.apply(self._init_weights)  # Initialize weights
        total_params = 0
        # Initialize attention projection layers and MLP down-sampling with a normal distribution
        for pname, p in self.named_parameters():
            total_params += p.numel()  # Also count parameters
            if pname.endswith('context_projection.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layer))

        print(f"Total model parameters: {total_params}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        position = torch.arange(0, T, dtype=torch.long, device=device)  # Position

        token_embed = self.transformer.token_embedding(idx)
        position_embed = self.transformer.position_embedding(position)

        x = self.transformer.dropout(token_embed + position_embed)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.norm(x)

        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pname: p for pname, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for pname, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for pname, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_no_decay = sum(p.numel() for p in no_decay_params)
        print(f"Parameters with weight decay: {num_decay}, without weight decay: {num_no_decay}")

        fused_avail = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avail and device_type == 'cuda'
        if use_fused:
            print("Using fused AdamW optimizer!")
        extra_args = {'fused': True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        return optimizer

    def generate(self, idx, max_generate_tokens, temperature=1.0, top_k=None):
        for _ in range(max_generate_tokens):
            idx = idx if idx.shape[1] <= self.args.block_size else idx[:, -self.args.block_size:]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
