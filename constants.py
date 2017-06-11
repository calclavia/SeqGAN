import string

# Model Parameters
MAX_VOCAB = 1024
EMBEDDING_DIM = 100
NUM_UNITS = 64

# Training Parameters
G = 2
SEQ_LEN = 128
TRAIN_WINDOW = 1
FAKE_GEN_BATCH_SIZE = 2048
NUM_FAKE = FAKE_GEN_BATCH_SIZE * 10
BATCH_SIZE = 256
ROLLOUT_BATCH = 8

# Generation Parameters
TEMP = 1
GEN_LEN = 64
BLEU_SAMPLES = 10
NGRAM = 4

# Paths
G_MODEL_PATH = 'out/generator.h5'
D_MODEL_PATH = 'out/discriminator.h5'

RL_G_MODEL_PATH = 'out/generator_rl.h5'
RL_D_MODEL_PATH = 'out/discriminator_rl.h5'

## TODO: New stuff

all_chars = string.ascii_letters + " .,;!?$#'\"-\n"
n_chars = len(all_chars)
batch_size = 32
g_units = 300
d_units = 300
