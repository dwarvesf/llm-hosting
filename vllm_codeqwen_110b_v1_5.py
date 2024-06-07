# # Fast inference with vLLM (Qwen/Qwen1.5-110B-Chat-AWQ)
#
# In this example, we show how to run basic inference, using [`vLLM`](https://github.com/vllm-project/vllm)
# to take advantage of PagedAttention, which speeds up sequential inferences with optimized key-value caching.

import os
import subprocess
import secrets


from modal import Image, Secret, App, enter, gpu, method, web_server

MODEL_DIR = "/model"
BASE_MODEL = "Qwen/Qwen1.5-110B-Chat-AWQ"

# ## Define a container image


# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Huggingface - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# Tip: avoid using global variables in this function. Changes to code outside this function will not be detected and the download step will not re-run.
def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()


# ### Image definition
# We'll start from a recommended Docker Hub image and install `vLLM`.
# Then we'll use `run_function` to run the function defined above to ensure the weights of
# the model are saved within the container image.
image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        "vllm==0.4.3",
        "wheel==0.43.0",
        "packaging==24.0",
        "huggingface_hub==0.23.3",
        "hf-transfer==0.1.6",
        "torch==2.3.0",
        "autoawq==0.2.5",
    )
    .apt_install("git")
    .run_commands(
        "pip install flash-attn==2.5.8 --no-build-isolation",
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secrets=[Secret.from_name("huggingface")],
        timeout=60 * 20,
    )
)

app = App("vllm-codeqwen-110b-v1.5", image=image)
GPU_CONFIG = gpu.A100(memory=80, count=2)


# Run a web server on port 8000 and expose vLLM OpenAI compatible server
@app.function(
    allow_concurrent_inputs=100,
    container_idle_timeout=60,
    gpu=GPU_CONFIG,
    secrets=[
        Secret.from_name("huggingface"),
        Secret.from_dotenv(),
    ],
)
@web_server(8000, startup_timeout=900)
def openai_compatible_server():
    target = BASE_MODEL
    cmd = f"python -m vllm.entrypoints.openai.api_server --model {target} --port 8000 --quantization awq"
    subprocess.Popen(cmd, shell=True)
