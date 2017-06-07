"""
Interprets an output using inverse index.
"""
import json
import argparse

def load_json_dict(fpath):
    with open(fpath, encoding='utf-8') as f:
        return json.loads(f.read())

parser = argparse.ArgumentParser(description='Parses output of model.')
parser.add_argument('file')

args = parser.parse_args()

# Load word index
word_index = load_json_dict('out/word_index.json')
inv_idx = {v: k for k, v in word_index.items()}

with open(args.file) as f:
    lines = f.readlines()

lines = [x.strip() for x in lines]
lines = [' '.join([inv_idx[int(word_id)] for word_id in l.split(' ')]) for l in lines]


with open(args.file + '.parse.txt', 'w') as f:
    f.write('\n'.join(lines))
