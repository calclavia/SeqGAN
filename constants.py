# Model Parameters
MAX_VOCAB = 1024
EMBEDDING_DIM = 100
NUM_UNITS = 256

# Training Parameters
SEQ_LEN = 20
NUM_FAKE = 1024 * 10
BATCH_SIZE = 128

# Generation Parameters
TEMP = .7
GEN_LEN = 64

# Paths
G_MODEL_PATH = 'out/generator.h5'
D_MODEL_PATH = 'out/discriminator.h5'
