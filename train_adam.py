import os
import time
import math
import pickle
import argparse  # Import argparse module
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Argument Parsing
parser = argparse.ArgumentParser(description="Train a GPT model.")
# I/O
parser.add_argument("--out_dir", type=str, default='out', help="Output directory for checkpoints and logs")
parser.add_argument("--eval_interval", type=int, default=2000, help="Interval between evaluations")
parser.add_argument("--log_interval", type=int, default=1, help="Interval between logging")
parser.add_argument("--eval_iters", type=int, default=200, help="Number of iterations to evaluate for")
parser.add_argument("--eval_only", action='store_true', help="Run evaluation only, then exit")
parser.add_argument("--always_save_checkpoint", action='store_true', help="Always save a checkpoint after each evaluation")
parser.add_argument("--init_from", type=str, default='scratch', choices=['scratch', 'resume', 'gpt2*'], help="Initialization method")
# wandb logging
parser.add_argument("--wandb_log", action='store_true', help="Enable wandb logging")
parser.add_argument("--wandb_project", type=str, default='owt', help="Wandb project name")
parser.add_argument("--wandb_run_name", type=str, default='gpt2', help="Wandb run name")
# data
parser.add_argument("--dataset", type=str, default='openwebtext', help="Dataset to use")
parser.add_argument("--gradient_accumulation_steps", type=int, default=5, help="Gradient accumulation steps")
parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
parser.add_argument("--block_size", type=int, default=1024, help="Block size for model input")
# model
parser.add_argument("--n_layer", type=int, default=12, help="Number of layers")
parser.add_argument("--n_head", type=int, default=12, help="Number of attention heads")
parser.add_argument("--n_embd", type=int, default=768, help="Embedding size")
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
parser.add_argument("--bias", action='store_true', help="Use bias in LayerNorm and Linear layers")
# optimizer
parser.add_argument("--optimizer_name", type=str, default='adamw', help="Optimizer type")
parser.add_argument("--learning_rate", type=float, default=6e-4, help="Learning rate")
parser.add_argument("--max_iters", type=int, default=600000, help="Maximum iterations")
parser.add_argument("--weight_decay", type=float, default=1e-1, help="Weight decay rate")
parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for optimizer")
parser.add_argument("--beta2", type=float, default=0.95, help="Beta2 for optimizer")
parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
parser.add_argument("--rho", type=float, default=0.1, help="Rho parameter for optimizer")
parser.add_argument("--interval", type=int, default=10, help="Interval parameter")
parser.add_argument("--variant", type=int, default=4, help="Variant parameter")
# learning rate decay settings
parser.add_argument("--decay_lr", action='store_true', help="Enable learning rate decay")
parser.add_argument("--warmup_iters", type=int, default=2000, help="Warmup iterations")
parser.add_argument("--lr_decay_iters", type=int, default=600000, help="Iterations for learning rate decay")
parser.add_argument("--lr_decay_rate", type=float, default=0.25, help="Minimum learning rate")
# DDP settings
parser.add_argument("--backend", type=str, default='nccl', help="Backend for distributed training")
# system
parser.add_argument("--device", type=str, default='cuda', help="Device for training")
parser.add_argument("--dtype", type=str, default='bfloat16', choices=['float32', 'bfloat16', 'float16'], help="Data type for training")
parser.add_argument("--compile", action='store_true', help="Compile the model for performance")
parser.add_argument("--scale_attn_by_inverse_layer_idx", action='store_true', help="Scale attention by inverse layer index")

parser.add_argument('--log_optimizer_state', action='store_true', default=False)
parser.add_argument("--state_interval", type=int, default=1, help="Interval to the previous optimizer state")
parser.add_argument("--log_weight_iters", type=str, default='None', help="Interval to the previous optimizer state")
parser.add_argument('--ckpo_with_current_time', action='store_true', default=False)

parser.add_argument('--wandb', action='store_false', default=True)

args = parser.parse_args()  # Parse arguments

try:
    os.environ["WANDB_HOST"] = os.environ.get('PJM_JOBID')
    args.job_id = os.environ.get('PJM_JOBID')
except:
    print('WANDB_HOST not set')

if args.log_weight_iters == 'None':
    args.log_weight_iters = []
else:
    args.log_weight_iters = args.log_weight_iters.split(',')

# Use parsed arguments
out_dir = args.out_dir
eval_interval = args.eval_interval
log_interval = args.log_interval
eval_iters = args.eval_iters
eval_only = args.eval_only
always_save_checkpoint = args.always_save_checkpoint
init_from = args.init_from
dataset = args.dataset
gradient_accumulation_steps = args.gradient_accumulation_steps
batch_size = args.batch_size
block_size = args.block_size
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
dropout = args.dropout
bias = args.bias
optimizer_name = args.optimizer_name
learning_rate = args.learning_rate
max_iters = args.max_iters
weight_decay = args.weight_decay
beta1 = args.beta1
beta2 = args.beta2
grad_clip = args.grad_clip
rho = args.rho
interval = args.interval
variant = args.variant
decay_lr = args.decay_lr
warmup_iters = args.warmup_iters
lr_decay_iters = args.lr_decay_iters
min_lr = args.learning_rate * args.lr_decay_rate
backend = args.backend
device = args.device
dtype = args.dtype
compile = args.compile
scale_attn_by_inverse_layer_idx = args.scale_attn_by_inverse_layer_idx
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
#exec(open('configurator.py').read()) # overrides from command line or config file
config = vars(args).copy()# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    gradient_accumulation_steps *= 8 # simulate 8 gpus

if master_process:
    if args.ckpo_with_current_time and args.wandb:
        import random, datetime
        dt_now = str(datetime.datetime.now()).replace(' ','-')
        args.out_dir=out_dir+'/'+dt_now
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(5000 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

if args.log_optimizer_state:
    param_name_dict = {}
    prev_weight_dict = {}
    prev_grad_dict = {}
    prev_momentum_dict = {}
    for name, param in model.named_parameters():
        param_name_dict[param] = name
    cos_func = torch.nn.CosineSimilarity(dim=0)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(optimizer_name, weight_decay, learning_rate, (beta1, beta2), rho, device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    del state_dict
    del checkpoint
# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def get_param_grad_norm(model):
    grad = {}
    weight = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad[name] = torch.norm(param.grad)
            weight[name] = torch.norm(param)
    return weight, grad

# logging
if args.wandb and master_process:
    import wandb
    wandb.init( config=config,
                entity=os.environ.get('WANDB_ENTITY', None),
                project=os.environ.get('WANDB_PROJECT', None),
                )

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
clip_time = 0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if args.wandb:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }, step=iter_num)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if str(iter_num) in args.log_weight_iters and master_process:
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': config,
        }
        print(f"saving checkpoint to {out_dir}")
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt_'+str(iter_num)+'.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if total_norm.item() > grad_clip:
            clip_time += 1
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    if iter_num % log_interval == 0 and master_process:
        if args.log_optimizer_state:
            state_info = {  'weight_norm/':{},'grad_norm/':{},'momentum_norm/':{},"weight_norm_element/":{},'grad_norm_element/':{},'momentum_norm_element/':{},
                            'weight_relative_error/':{},'weight_relative_error_element/':{},'weight_cosine_sim/':{},'weight_norm_ratio/':{},'weight_norm_element_ratio/':{},
                            'grad_relative_error/':{},'grad_relative_error_element/':{},'grad_cosine_sim/':{},'grad_norm_ratio/':{},'grad_norm_element_ratio/':{},
                            'momentum_relative_error/':{},'momentum_relative_error_element/':{},'momentum_cosine_sim/':{},'momentum_norm_ratio/':{},'momentum_norm_element_ratio/':{},
                            'update_relative_error/':{},'update_relative_error_element/':{},'update_cosine_sim/':{},'update_norm_ratio/':{},'update_norm_element_ratio/':{}}
            for p in optimizer.state.keys():
                param_name = param_name_dict[p]
                weight = p.data
                grad = p.grad
                momentum = optimizer.state[p]["exp_avg"]
                state_info['weight_norm/'][param_name] = torch.norm(p.data).item()
                state_info['grad_norm/'][param_name] = torch.norm(p.grad).item()
                state_info['momentum_norm/'][param_name] = torch.norm(momentum).item()
                state_info['weight_norm_element/'][param_name] = torch.abs(p.data).mean(dtype=torch.float32).item()
                state_info['grad_norm_element/'][param_name] = torch.abs(p.grad).mean(dtype=torch.float32).item()
                state_info['momentum_norm_element/'][param_name] = torch.abs(momentum).mean(dtype=torch.float32).item()

                if param_name in prev_weight_dict.keys():
                    prev_weight = prev_weight_dict[param_name]
                    state_info['weight_relative_error/'][param_name] = torch.norm(weight - prev_weight).item() / state_info['weight_norm/'][param_name]
                    state_info['weight_relative_error_element/'][param_name] = torch.abs(grad - prev_weight).mean(dtype=torch.float32).item() / state_info['weight_norm_element/'][param_name]
                    state_info['weight_cosine_sim/'][param_name] = cos_func(weight.view(-1), prev_weight.view(-1))
                    state_info['weight_norm_ratio/'][param_name] = state_info['weight_norm/'][param_name] / torch.norm(prev_weight).item()
                    state_info['weight_norm_element_ratio/'][param_name] = state_info['weight_norm_element/'][param_name] / torch.abs(prev_weight).mean(dtype=torch.float32).item()
                if param_name in prev_grad_dict.keys():
                    prev_grad = prev_grad_dict[param_name]
                    state_info['grad_relative_error/'][param_name] = torch.norm(grad - prev_grad).item() / state_info['grad_norm/'][param_name]
                    state_info['grad_relative_error_element/'][param_name] = torch.abs(grad - prev_grad).mean(dtype=torch.float32).item() / state_info['grad_norm_element/'][param_name]
                    state_info['grad_cosine_sim/'][param_name] = cos_func(grad.view(-1), prev_grad.view(-1))
                    state_info['grad_norm_ratio/'][param_name] = state_info['grad_norm/'][param_name] / torch.norm(prev_grad).item()
                    state_info['grad_norm_element_ratio/'][param_name] = state_info['grad_norm_element/'][param_name] / torch.abs(prev_grad).mean(dtype=torch.float32).item()
                if param_name in prev_momentum_dict.keys():
                    prev_momentum = prev_momentum_dict[param_name]
                    state_info['momentum_relative_error/'][param_name] = torch.norm(momentum - prev_momentum) / state_info['momentum_norm/'][param_name]
                    state_info['momentum_relative_error_element/'][param_name] = torch.abs(momentum - prev_momentum).mean(dtype=torch.float32).item() / state_info['momentum_norm_element/'][param_name]
                    state_info['momentum_cosine_sim/'][param_name] = cos_func(momentum.view(-1), prev_momentum.view(-1))
                    state_info['momentum_norm_ratio/'][param_name] = state_info['momentum_norm/'][param_name]/ torch.norm(prev_momentum).item()
                    state_info['momentum_norm_element_ratio/'][param_name] = state_info['momentum_norm_element/'][param_name] / torch.abs(prev_momentum).mean(dtype=torch.float32).item()
                if param_name in prev_grad_dict.keys() and param_name in prev_momentum_dict.keys():
                    update = momentum.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
                    prev_update = prev_momentum.clone().mul_(beta1).add(prev_grad, alpha=1 - beta1).sign_()
                    state_info['update_relative_error/'][param_name] = torch.norm(update - prev_update) / torch.norm(update)
                    state_info['update_relative_error_element/'][param_name] = torch.abs(update - prev_update).mean(dtype=torch.float32).item() / torch.abs(update).mean(dtype=torch.float32).item()
                    state_info['update_cosine_sim/'][param_name] = cos_func(update.view(-1), prev_update.view(-1))
                    state_info['update_norm_ratio/'][param_name] = torch.norm(update)/ torch.norm(prev_update).item()
                    state_info['update_norm_element_ratio/'][param_name] = torch.abs(update).mean(dtype=torch.float32).item() / torch.abs(prev_update).mean(dtype=torch.float32).item()

    # Save Previous Information
    if iter_num % log_interval == log_interval-args.state_interval and args.log_optimizer_state and master_process and master_process:
        if args.log_optimizer_state:
            for p in optimizer.state.keys():
                param_name = param_name_dict[p]
                prev_weight_dict[param_name] = p.data.detach().clone()
                prev_grad_dict[param_name] = p.grad.detach().clone()
                prev_momentum_dict[param_name] = optimizer.state[p]["exp_avg"].detach().clone()

    if str(iter_num) in args.log_weight_iters and master_process:
        if args.wandb:
            log_dict = {
                "iter": iter_num,
            }
            for p in optimizer.state.keys():
                param_name = param_name_dict[p]
                grad = p.grad
                log_dict['gradient/'+param_name] = wandb.Histogram(grad.cpu().detach().numpy())
                if "exp_avg" in optimizer.state[p]:
                    momentum = optimizer.state[p]["exp_avg"]
                    log_dict['momentum/'+param_name] = wandb.Histogram(momentum.cpu().detach().numpy())
                if "momentum" in optimizer.state[p]:
                    momentum = optimizer.state[p]["momentum"]
                    log_dict['momentum/'+param_name] = wandb.Histogram(momentum.cpu().detach().numpy())
            wandb.log(log_dict, step=iter_num)
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        params = []
        for (name, p) in model.named_parameters():
            params.append(p)
        total_param_norm = 0
        for p in params:
            param_norm = p.data.norm(2)
            total_param_norm += param_norm.item() ** 2
        total_param_norm = total_param_norm ** 0.5
        momentum_norm = 0
        LL = len(optimizer.state_dict()['state'])
        for jj in range(LL):
            if 'exp_avg' in optimizer.state_dict()['state'][jj]:
                momentum_norm += (optimizer.state_dict()['state'][jj]['exp_avg'].detach().norm(2)) ** 2
            if 'momentum' in optimizer.state_dict()['state'][jj]:
                momentum_norm += (optimizer.state_dict()['state'][jj]['momentum'].detach().norm(2)) ** 2
        momentum_norm = torch.sqrt(momentum_norm).item()
        if args.wandb:
            log_dict = {
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "param_norm": total_param_norm,
                "momentum_norm" : momentum_norm,
                "train/clip_rate": clip_time / (iter_num + 1),
            }
            if args.log_optimizer_state:
                log_dict.update(state_info)
            wandb.log(log_dict, step=iter_num)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
