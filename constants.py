# Model Parameters
MAX_VOCAB = 1024
EMBEDDING_DIM = 100
NUM_UNITS = 256

# Training Parameters
SEQ_LEN = 32
TRAIN_WINDOW = 1
FAKE_GEN_BATCH_SIZE = 2048
NUM_FAKE = FAKE_GEN_BATCH_SIZE * 10
BATCH_SIZE = 256
ROLLOUT_BATCH = 32

# Generation Parameters
TEMP = 1
GEN_LEN = 64

# Paths
G_MODEL_PATH = 'out/generator.h5'
D_MODEL_PATH = 'out/discriminator.h5'

RL_G_MODEL_PATH = 'out/generator_rl.h5'
RL_D_MODEL_PATH = 'out/discriminator_rl.h5'
