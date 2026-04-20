"""
IatrogeniX — upload_to_hf.py
==============================
Automated deployment to Hugging Face.
Creates Model Repo and Space, then uploads weights and demo app.

Requirements:
  pip install huggingface_hub
"""
import os
import argparse
from huggingface_hub import HfApi, create_repo, login

def upload(token: str, username: str, repo_name: str, private: bool):
    api = HfApi(token=token)
    
    # 1. Create Model Repository
    model_repo_id = f"{username}/{repo_name}"
    print(f"Creating/Checking Model Repo: {model_repo_id}...")
    create_repo(model_repo_id, repo_type="model", private=private, exist_ok=True)
    
    # 2. Upload Model weights (GGUF)
    gguf_path = "models/iatrogenix-q5_k_m.gguf"
    if os.path.exists(gguf_path):
        print(f"Uploading GGUF weights (this may take a while)...")
        api.upload_file(
            path_or_fileobj=gguf_path,
            path_in_repo="models/iatrogenix-q5_k_m.gguf",
            repo_id=model_repo_id,
            repo_type="model"
        )
    
    # 3. Upload Model Card
    readme_path = "docs/hf_readme.md"
    if os.path.exists(readme_path):
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=model_repo_id,
            repo_type="model"
        )
        
    # 4. Create and Setup Space
    space_repo_id = f"{username}/{repo_name}-Demo"
    print(f"Creating/Checking Space: {space_repo_id}...")
    create_repo(space_repo_id, repo_type="space", space_sdk="gradio", private=private, exist_ok=True)
    
    # Upload App files to Space
    files_to_upload = {
        "app/app.py": "app.py",
        "requirements_hf.txt": "requirements.txt",
        "safety/validator.py": "safety/validator.py",
        "safety/drugs.json": "safety/drugs.json",
        "safety/protocols.json": "safety/protocols.json",
        "docs/hf_readme.md": "README.md"
    }
    
    for local, remote in files_to_upload.items():
        if os.path.exists(local):
            print(f"Uploading {local} to Space...")
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=space_repo_id,
                repo_type="space"
            )

    print(f"\nDeployment Complete!")
    print(f"Model: https://huggingface.co/{model_repo_id}")
    print(f"Space: https://huggingface.co/spaces/{space_repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IatrogeniX HF Uploader")
    parser.add_argument("--token", required=True, help="HF Write Token")
    parser.add_argument("--username", required=True, help="HF Username")
    parser.add_argument("--repo", default="IatrogeniX", help="Repo name")
    parser.add_argument("--public", action="store_true", help="Make repo public")
    
    args = parser.parse_args()
    upload(args.token, args.username, args.repo, not args.public)
