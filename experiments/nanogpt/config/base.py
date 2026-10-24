# Base shape d=256, L=2, N=1 (m_n = m_L = 1): tune the base learning rate eta here.
n_embd = 256
n_head = 4
n_layer = 2
n_loops = 1
train_tokens = 11781 * 2 ** 18
batch_tokens = 2 ** 18
micro_batch = 64
learning_rate = 2 ** -8
