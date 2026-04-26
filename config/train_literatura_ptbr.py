# DuckGPT — train a small BPE-level model on PT-BR Project Gutenberg literature.
#
# Prereqs:
#   1. Train a BPE tokenizer + prepare data
#        python data/literatura_ptbr/prepare.py
#      (writes models/bpe.* and data/literatura_ptbr/{train,val}.bin)
#
# Run:
#        python lib/train.py config/train_literatura_ptbr.py

out_dir = 'out-literatura-ptbr'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'duckgpt-ptbr'
wandb_run_name = 'literatura'

dataset = 'literatura_ptbr'      # data/literatura_ptbr/{train,val}.bin
gradient_accumulation_steps = 4
batch_size = 32
block_size = 256

# small but useful model — fits on a 6 GB GPU at bf16
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False

learning_rate = 3e-4
max_iters = 8000
lr_decay_iters = 8000
min_lr = 3e-5
warmup_iters = 200
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
compile = False        # set True if PyTorch 2.x + GPU
