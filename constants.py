import string

# Model Parameters
ALL_CHARS = string.ascii_letters + " .,;!?$#'\"-\n"
N_CHARS = len(ALL_CHARS)

G_UNITS = 512
D_UNITS = 512

# Training Parameters
SEQ_LEN = 128
BATCH_SIZE = 32
MAX_ITERATIONS = 100000
GAN_ITERATIONS = 10000
D_STEPS = 5

# Generation Parameters
TEMP = 1
BLEU_SAMPLES = 10
NGRAM = 4

# Paths
G_MODEL_PATH = 'out/generator.torch'
D_MODEL_PATH = 'out/discriminator.torch'
