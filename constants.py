# Model Parameters
MAX_VOCAB = 1024
EMBEDDING_DIM = 100
NUM_UNITS = 256

# Training Parameters
SEQ_LEN = 20
TRAIN_WINDOW = 3
FAKE_GEN_BATCH_SIZE = 2048
NUM_FAKE = FAKE_GEN_BATCH_SIZE * 10
BATCH_SIZE = 256

# Generation Parameters
TEMP = .7
GEN_LEN = 64

# Paths
G_MODEL_PATH = 'out/generator.h5'
D_MODEL_PATH = 'out/discriminator.h5'
