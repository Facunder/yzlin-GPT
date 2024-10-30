import os
import requests
import tiktoken
import numpy as np
import hashlib

# *To solve the http timeout problem of server links, use pre-downloaded files*
tiktoken_cache_dir = "./.tiktoken/"
blobpath_1 = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"
blobpath_2 = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json"
cache_key_1 = hashlib.sha1(blobpath_1.encode()).hexdigest()
cache_key_2 = hashlib.sha1(blobpath_2.encode()).hexdigest()
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
assert os.path.exists(os.path.join(tiktoken_cache_dir, cache_key_1))
assert os.path.exists(os.path.join(tiktoken_cache_dir, cache_key_2))

input_file_path = os.path.join(os.path.dirname(__file__), 'datasets/TinyStories-cut.txt')
# train has 135,808,280 tokens
# val has 15,126,332 tokens

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")

train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'datasets/datrain.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'datasets/val.bin'))