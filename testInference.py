import os
import tiktoken
import torch
from model import GPT,Model_args
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

checkpoint_save_dir = './checkpoints'
device = 'cuda'
device_type = 'cuda'
dtype = 'bfloat16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# generate参数
top_k = 30
tempreture = 0.3 # bigger means more random
num_samples = 1 # sample nums
max_new_tokens = 128

# load checkpoint
print(f"load checkpoint from {checkpoint_save_dir}")
ckpt_path = os.path.join(checkpoint_save_dir,'checkpoint.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
args = checkpoint['model_args']
model = GPT(Model_args(**args))
state_dict = checkpoint['model']

# according to NanoGPT debug
unwanted_prefix = '_orig_mod'
for k,v in list(state_dict.items()): 
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
 
model.eval()
model.to(device)

# Prevent "<|endoftext|>" from being incorrectly tokenized
enc = tiktoken.get_encoding("gpt2")# gpt2 tokenizer
decode = lambda x:enc.decode(x)
encode = lambda x:enc.encode(x,allowed_special={"<|endoftext|>"}) 

start = "I am a human in" # start text
start_ids = encode(start)
x = torch.tensor(start_ids,dtype=torch.long,device=device).unsqueeze(0)

ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x,max_new_tokens,top_k=top_k,tempreture=tempreture)
            print(decode(y[0].tolist()))
            print("----------")