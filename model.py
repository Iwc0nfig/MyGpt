import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
import inspect
import time
import os
import numpy as np

torch.cuda.empty_cache()
torch.cuda.ipc_collect()
torch.set_float32_matmul_precision('high')

class DataloaderLite:
    def __init__(self, B, T, data_dir="./tokenized_data"):
        self.B = B
        self.T = T
        self.data_dir = data_dir

        # Get a sorted list of all .npy files in the data directory
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')])
        if not self.files:
            raise ValueError(f"No .npy files found in {data_dir}")

        self.current_file_idx = 0
        self.tokens = np.array([], dtype=np.uint16)
        self.current_position = 0
        
        # Initial load of tokens from the first file
        self._load_next_chunk()

        print(f"Dataloader initialized. Found {len(self.files)} token chunk files.")
        print(f"Total tokens in buffer after initial load: {len(self.tokens)}")

    def _load_next_chunk(self):
        """
        Loads the next .npy file into the token buffer.
        If all files have been processed, it loops back to the beginning for the next epoch.
        """
        # If we have processed all files, loop back to the beginning
        if self.current_file_idx >= len(self.files):
            self.current_file_idx = 0
            print("All token chunks processed. Resetting to the beginning for the next epoch.")

        # Load the next .npy file
        file_path = self.files[self.current_file_idx]
        # Allow pickle loading since the dtype is object
        loaded_chunks = np.load(file_path, allow_pickle=True)
        
        # Concatenate all token arrays in the chunk into a single flat array
        new_tokens = np.concatenate(loaded_chunks)
        
        # Append the new tokens to our existing buffer
        self.tokens = np.concatenate((self.tokens, new_tokens))
        
        self.current_file_idx += 1
        print(f"Loaded {file_path}. Total tokens in buffer: {len(self.tokens)}")


    def next_batch(self):
        """
        Retrieves the next batch of data. If the buffer runs low, it loads the next file chunk.
        """
        B, T = self.B, self.T
        
        # Check if we have enough tokens for a full batch. If not, load the next chunk.
        # This loop ensures we have enough data, even if a single chunk is smaller than a batch.
        while len(self.tokens) < self.current_position + B * T + 1:
            print("Buffer running low, attempting to load next chunk...")
            
            # Check if there are more files to load
            if self.current_file_idx < len(self.files):
                # We have more files, so we can try to load more tokens
                # First, we trim the tokens that are already processed
                self.tokens = self.tokens[self.current_position:]
                self.current_position = 0
                self._load_next_chunk()
            else:
                # No more files to load, and buffer is still not enough. Resetting.
                print("Warning: Reached end of dataset and not enough tokens for a full batch. Resetting position.")
                self.current_position = 0
                # If after reset we still don't have enough, we're stuck in a loop.
                if len(self.tokens) < B * T + 1:
                    raise ValueError("Dataset is too small to form even one batch. Check your data or batch/sequence size.")


        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int64), dtype=torch.long) # Convert to torch tensor
        
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T

        return x, y
        

class LayerNornm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()

        self.weights = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self,input):
        return F.layer_norm(input, self.weights.shape , self.weights, self.bias, eps=1e-5)
    

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd , 3*config.n_embd  ,bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_haed = config.n_head #12
        self.n_embd = config.n_embd #768
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional,'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
            
    
            
    def forward(self,x):
        B,T,C = x.size()

        q,k,v = self.c_attn(x).split(self.n_embd,dim=2) #split the 3 dimension  because we did that 3*n+embd
        # q,k,v shape is (B,T,C) where C = n_embd
        # We will transpose the k,q,v from [B, T, n_head,head_dim] to [B, n_head, T , head_dim]
        k = k.view(B,T,self.n_haed,C // self.n_haed).transpose(1,2)
        q = q.view(B,T,self.n_haed,C // self.n_haed).transpose(1,2)
        v = v.view(B,T,self.n_haed,C // self.n_haed).transpose(1,2)

        #Compute attention scores → apply softmax → optionally mask → apply dropout → multiply by values.
        
        y = torch.nn.functional.scaled_dot_product_attention(q,k,v , 
                                                             attn_mask=None,
                                                             dropout_p=self.dropout if self.training else 0 , 
                                                             is_causal= True)
        
        
       
        y =y.transpose(1,2).contiguous().view(B,T,C)
        y = self.resid_dropout(y)
        return y
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNornm(config.n_embd,config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNornm(config.n_embd,config.bias)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
@dataclass
class Config:
    block_size: int = 2048
    vocab_size: int = 16384
    n_layers: int = 12
    n_head: int = 12
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.config = config
        #final_embedding = token_embedding + position_embedding
        self.transformer = nn. ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd), #word token embedding 
            wpe = nn.Embedding(config.block_size,config.n_embd), #word position embedding 
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = LayerNornm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
    
    # this is a helper function to initialize the weights of the model based on gpt-2 paper
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            #cross-entry-loss can't take multidimensional targets, so we need to reshape them to flattened vectors to 2d
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


total_batch_size = 262_144 # 2**18
B =16
T = 1024
grad_accum_steps = total_batch_size // (B * T)  # how many steps to accumulate gradients before updating the model
print(f"Total desired batch size: {total_batch_size} \n grad_accum_steps: {grad_accum_steps}")
    
# Initialize the dataloader with the directory of tokenized .npy files
train_loader = DataloaderLite(B=B, T=T, data_dir="./tokenized_data")
model = GPT(Config())
model.eval()
device = 'cuda'
model.to(device)
#model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr*0.1
warmup_steps = 1_500
max_steps = 19_073

print(f"Model is running on device: {device}")

torch.manual_seed(1337)  # for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)  # for reproducibility on GPU


def get_lr(it):
    #linear warmup 
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > warmup_steps:
        return min_lr
    #in between warmup and max_steps we use cosine decay
    decay_ration = (it-warmup_steps)/(max_steps-warmup_steps)
    assert 0<=decay_ration <= 1, "Decay ratio should be between 0 and 1"
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ration))  # cosine decay
    return min_lr + coeff * (max_lr - min_lr)  # linear decay from min_lr to max_lr

#optimer
optimizer = model.configure_optimizers(weight_decay=1e-4,learning_rate=max_lr, betas=(0.9, 0.95), device_type=device)



for step in range(max_steps):
    t0 = time.time()
      # Move data to the same device as the model
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x,y= train_loader.next_batch()
        x,y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):   
            logits , loss = model(x, y)
        # accumulate gradients
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
         
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping

    optimizer.step()
    torch.cuda.synchronize() 
    t1 = time.time()
    dt = (t1-t0)*1000_000
    tokens_per = (train_loader.B * train_loader.T) * grad_accum_steps 
    tokens_per_sec = tokens_per / (t1-t0)  # tokens per second
    print(f"Step {step} | Loss: {loss_accum.item():.4f}| lr : {lr:.6f} | norm: {norm:.2f} | dt: {dt:.2f} s |  tokens/sec: {tokens_per_sec:.2f}")



def save_model(full_model = False):
    if full_model:
        torch.save(model, "full_model.pth")
        print("Entire model saved successfully!") #when you load it you will need to evaluate the model (model.eval())
    else:
        torch.save(model.state_dict(), "model.pth") #To use it you will need ot have the class Net in your code 
        print("Model saved successfully!")
        
save_model(full_model=True) # save the entire model