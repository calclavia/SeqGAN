import torch

from util import *
from p_model import *
from constants import *
from data_loader import *

def main():
    torch.backends.cudnn.enabled = False

    print('Building models...')
    common = CommonModule(n_chars, G_UNITS)
    generator = Generator(n_chars, G_UNITS, common).cuda()

    try:
        generator.load_state_dict(torch.load(G_MODEL_PATH))
        print('Loaded generator model')
    except Exception as e:
        print('Unable to load generator', e)

    print('Generating...')
    print(generator.sample()[0][0])

if __name__ == '__main__':
    main()
