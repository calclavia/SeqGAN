An implementation of SeqGAN in Pytorch.

## Requirements
- Python 3.6
- Pytorch (http://pytorch.org/)
- TQDM

To install all Python dependencies:
```
pip install -r requirements.txt
```

### Training
To train the model, run:

```
python train.py
```

### Generating
To use a trained model to generate text, run:

```
python generate.py
```

The saved model file must be located in `out/generator.torch`.
