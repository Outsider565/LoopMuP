# Looped shape d=512, L=4, N=4 under Loop-μP; set --learning_rate to the base-shape optimum.
n_embd = 512
n_head = 8
n_layer = 4
n_loops = 4
alpha_N = 1.0
alpha_L = 0.5
train_tokens = 11781 * 2 ** 18
batch_tokens = 2 ** 18
micro_batch = 16
learning_rate = 2 ** -8
