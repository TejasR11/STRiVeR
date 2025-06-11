import os
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# Set the cache directory to our workspace
os.environ['TRANSFORMERS_CACHE'] = '/workspace/models/cache'
os.environ['HF_HOME'] = '/workspace/models/cache'

print("Downloading Qwen2.5-Omni-3B model...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-Omni-3B',
    torch_dtype='auto',
    device_map='auto',
    cache_dir='/workspace/models/cache'
)
print("Downloading processor...")
processor = Qwen2_5OmniProcessor.from_pretrained(
    'Qwen/Qwen2.5-Omni-3B',
    cache_dir='/workspace/models/cache'
)
print("Download complete!") 