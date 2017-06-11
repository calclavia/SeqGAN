import string

# Model Parameters
all_chars = string.ascii_letters + " .,;!?$#'\"-\n"
n_chars = len(all_chars)

G_UNITS = 512
D_UNITS = 512

# Training Parameters
SEQ_LEN = 128
BATCH_SIZE = 32
max_iterations = 100000
gan_iteration = 10000

# Generation Parameters
TEMP = 1
BLEU_SAMPLES = 10
NGRAM = 4

# Paths
G_MODEL_PATH = 'out/generator.torch'
D_MODEL_PATH = 'out/discriminator.torch'

RL_G_MODEL_PATH = 'out/generator_rl.h5'
RL_D_MODEL_PATH = 'out/discriminator_rl.h5'
