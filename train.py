import os
import numpy as np
import torch
import torch.nn as nn
import math
from model import ModelArgs, yzlinGPT
import time

# Model parameters
block_size = 128  # Block size for GPT2 is typically 1024, can not afford that much memory
batch_size = 64  # Tentative, will adjust based on memory usage
n_layer = 8
n_head = 8
n_embed = 768
bias = False
dropout = 0.0
dataset_path = './datasets'
print("dataset_path: ", dataset_path)
init_from = 'scratch'  # 'scratch' or 'resume' - start training from scratch or resume
checkpoint_save_dir = './checkpoint'
eval_iters = 200
log_interval = 10
eval_interval = 1000  # Evaluate and save checkpoint every n steps
# Learning rate decay
learning_rate = 6e-4
warmup_iters = 2000
lr_decay_iters = 8000
min_lr = 6e-5
gradient_accum_steps = 5 * 2
# Optimizer parameters
max_iters = 600000  # Train for a number of iterations
weight_decay = 1e-1
betas = (0.9, 0.95)
grad_clip = 1.0  # Gradient clipping
# System settings
device = 'cuda:3'
device_type = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# DataLoader
data_dir = os.path.join(dataset_path)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    # print(split, 'data shape:', data.shape)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size].astype(np.int64))) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size].astype(np.int64))) for i in ix])

    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

model_args = dict(n_layer=n_layer, n_head=n_head, n_embed=n_embed, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

iter_num = 0
best_val_loss = 1e9

assert init_from in ['scratch', 'resume']
if init_from == 'scratch':
    print("Starting training from scratch")
    model_args['vocab_size'] = 50304
    model = yzlinGPT(ModelArgs(**model_args))

elif init_from == 'resume':
    print("Resuming training")
    ckpt_path = os.path.join(checkpoint_save_dir, 'checkpoint.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    for key in ['n_layer', 'n_head', 'n_embed', 'block_size', 'bias', 'vocab_size']:
        model_args[key] = checkpoint['model_args'][key]
    model = yzlinGPT(ModelArgs(**model_args))
    model.load_state_dict(checkpoint['model'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
model.to(device)
optimizer = model.configure_optimizers(weight_decay, learning_rate, betas, device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # Clear checkpoint after loading

def estimate_loss():
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        loss_values = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            loss_values[i] = loss.item()
        losses[split] = loss_values.mean().item()
    model.train()
    return losses

def get_lr(current_iter):
    if current_iter < warmup_iters:
        return learning_rate * current_iter / warmup_iters
    elif current_iter > lr_decay_iters:
        return min_lr
    else:
        rate = (current_iter - warmup_iters) / (lr_decay_iters - warmup_iters)
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * rate)) * (learning_rate - min_lr)

X, Y = get_batch('train')
t_start = time.time()
while True:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if iter_num > 0 and iter_num % eval_interval == 0:
        loss_report = estimate_loss()
        print(f"Iteration {iter_num}, Train loss: {loss_report['train']}, Validation loss: {loss_report['val']}")
        best_val_loss = min(loss_report['val'], best_val_loss)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, os.path.join(checkpoint_save_dir, 'checkpoint.pt'))
        print(f"Checkpoint saved at {checkpoint_save_dir}/checkpoint.pt")
    for micro_step in range(gradient_accum_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accum_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    if grad_clip > 0.0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if iter_num > 0 and iter_num % log_interval == 0:
            print(f"iter:{iter_num},loss:{loss.item()*gradient_accum_steps}")

    # t_end = time.time()
    iter_num += 1
    if iter_num > max_iters:
        break
