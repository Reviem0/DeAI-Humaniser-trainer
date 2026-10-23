import os
for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"]:
    print(k, "=", os.environ.get(k))

from huggingface_hub import constants
print("Hub cache in use:", constants.HF_HUB_CACHE)