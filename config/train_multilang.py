# DuckGPT — train on the trilingual corpus:
#   - English encyclopedic (WikiText-103)
#   - Portuguese encyclopedic (Wikipedia PT-BR 20231101)
#   - EN↔PT dialogue / "falas" (OPUS-100 en-pt: subtitles + EuroParl + books)
#
# Prereqs:
#   1. Train BPE tokenizer
#        python lib/tokenizer.py train --input data/literatura_ptbr/raw \
#            --vocab-size 16000 --out models/bpe
#   2. Build the dataset
#        python data/multilang_en_pt/prepare.py
#
# Run:
#        python lib/train.py config/train_multilang.py

out_dir = 'out-multilang'
eval_interval = 500
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'duckgpt-multilang'
wandb_run_name = 'enpt-encyclopedic-dialogue'

dataset = 'multilang_en_pt'          # data/multilang_en_pt/{train,val}.bin
gradient_accumulation_steps = 8       # effective batch ~= 8 * batch_size
batch_size = 32
block_size = 512                      # longer context for dialogue pairs

# small-medium model — useful on a 12 GB GPU at bf16
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

learning_rate = 3e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 3e-5
warmup_iters = 500
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
compile = False        # set True if PyTorch 2.x + GPU
