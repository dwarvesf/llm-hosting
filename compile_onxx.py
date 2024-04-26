import os
import subprocess
import secrets


from modal import Image, Secret, Stub, enter, gpu, method, web_server

MODEL_DIR = "/model"
BASE_MODEL = "Snowflake/snowflake-arctic-embed-l"
huggingface_secret = Secret.from_name("huggingface")

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
        "optimum[exporters,onnxruntime]==1.19.1",
        "huggingface_hub==0.22.2",
        "hf-transfer==0.1.6",
        "torch==2.2.1",
        "transformers==4.40.1",
        "sentence-transformers==2.7.0",
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secrets=[huggingface_secret],
        timeout=60 * 20,
    )
)

stub = Stub("optimum-onxx-compile", image=image)
GPU_CONFIG = gpu.A100(memory=40, count=1)

@stub.function(
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
)
def compile_onnx():
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    import huggingface_hub

    # Get model from checkpoint
    model_checkpoint = "Snowflake/snowflake-arctic-embed-l"
    save_directory = "onnx/"

    # Load a model from Sentence Transformers and export it to ONNX
    ort_model = ORTModelForFeatureExtraction.from_pretrained(model_checkpoint, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Save the onnx model and tokenizer
    ort_model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    # Upload the model to Hugging Face
    api = huggingface_hub.HfApi(token=os.getenv("HF_TOKEN"))
    folder = huggingface_hub.HfFolder()
    token = folder.get_token()
    try:
        api.create_repo(repo_id="monotykamary/snowflake-arctic-embed-l-onnx")
    except:
        pass
    api.upload_file(
        token=token,
        repo_id=f"monotykamary/snowflake-arctic-embed-l-onnx",
        path_or_fileobj=save_directory,
        path_in_repo="onnx",
    )

@stub.local_entrypoint()
def main():
    compile_onnx.remote()