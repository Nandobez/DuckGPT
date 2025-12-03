# Smoke test config: char-level Shakespeare, 100 iters, tiny model.
# Designed to run end-to-end on CPU in ~2 minutes.

out_dir = 'out-smoke'
eval_interval = 50
eval_iters = 20
log_interval = 10
always_save_checkpoint = True

wandb_log = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128

# very small model
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 3e-3
max_iters = 100
lr_decay_iters = 100
min_lr = 1e-4
warmup_iters = 10
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
compile = False
