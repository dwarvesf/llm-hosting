import os
import shutil
import fnmatch
from pydantic import BaseModel
from fastapi import HTTPException, Header
from fastapi.responses import JSONResponse
from modal import Image, App, web_endpoint, Secret
from typing import Optional, List
from enum import Enum

# Create Modal Image with required dependencies
image = (
    Image.debian_slim()
    .apt_install("git")
    .pip_install("gitpython")
)

# Create Modal App
app = App(name="git-traverser")

class RepoType(str, Enum):
    GITHUB = "github"
    GITLAB = "gitlab"

class GitRepoRequest(BaseModel):
    repo_url: str
    branch: Optional[str] = "main"
    type: Optional[RepoType] = None
    file_patterns: Optional[List[str]] = None

# Directories and files to ignore
IGNORE_PATTERNS = [
    "node_modules",
    "__pycache__",
    "env",
    "venv",
    ".venv",
    "virtualenv",
    "target/dependency",
    "build/dependencies",
    "dist",
    "out",
    "bundle",
    "vendor",
    "tmp",
    "temp",
    "deps",
    "pkg",
    "Pods",
    ".git",
    ".*",
    "*.lock",  # This will catch package-lock.json, yarn.lock, Gemfile.lock, etc.
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Gemfile.lock",
    "Pipfile.lock",
    "poetry.lock",
    "composer.lock",
    "Cargo.lock",
    "mix.lock",
    "shard.lock",
    "Podfile.lock",
    "gradle.lockfile",
    "pubspec.lock",
    "project.assets.json",
    "packages.lock.json",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.exe",
    "*.bin",
    "*.obj",
    "*.o",
    "*.a",
    "*.lib",
    "*.log",
    "*.cache",
    "*.bak",
    "*.swp",
    "*.swo",
    "*.tmp",
    "*.temp",
    "*.DS_Store",
    "Thumbs.db",
    "desktop.ini",
    "go.sum",
]

# Important file patterns
DEFAULT_IMPORTANT_FILE_PATTERNS = [
    "*.md",
    "README*",
    "CONTRIBUTING*",
    "CHANGELOG*",
    "go.mod",
    "go.sum",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "Gemfile",
    "Gemfile.lock",
    "requirements.txt",
    "setup.py",
    "Pipfile",
    "Pipfile.lock",
    "pom.xml",
    "build.gradle",
    "Cargo.toml",
    "Cargo.lock",
    "devbox.json",
    "Dockerfile",
    ".gitignore",
    ".dockerignore",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".env.example",
    "Makefile",
    "*.config.js",
    "tsconfig.json",
    "tslint.json",
    "eslintrc.*",
    "prettierrc.*",
]

def should_ignore(path: str) -> bool:
    name = os.path.basename(path)
    return any(fnmatch.fnmatch(name, pattern) for pattern in IGNORE_PATTERNS)

def is_important_file(filename: str, custom_patterns: Optional[List[str]] = None) -> bool:
    patterns = custom_patterns if custom_patterns is not None else DEFAULT_IMPORTANT_FILE_PATTERNS
    return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)

def validate_bearer_token(bearer_token: str, valid_token: str) -> bool:
    return bearer_token == f"Bearer {valid_token}"

def detect_repo_type(repo_url: str) -> RepoType:
    if "github.com" in repo_url:
        return RepoType.GITHUB
    elif "gitlab.com" in repo_url:
        return RepoType.GITLAB
    else:
        raise ValueError("Unable to detect repository type. Please specify 'type' in the request.")

@app.function(image=image)
def traverse_git_repo(repo_url: str, branch: str = "main", repo_type: RepoType = None, token: Optional[str] = None, file_patterns: Optional[List[str]] = None) -> dict:
    """
    Clone a git repository, traverse it, and return its directory structure.
    """
    import git
    from urllib.parse import urlparse

    # Detect repo type if not provided
    if repo_type is None:
        repo_type = detect_repo_type(repo_url)

    # Extract repo name from the URL
    repo_name = os.path.splitext(os.path.basename(urlparse(repo_url).path))[0]
    clone_dir = f"/tmp/{repo_name}"

    def attempt_clone(branch):
        try:
            # Prepare clone options
            clone_options = {'branch': branch}
            if token:
                if repo_type == RepoType.GITHUB:
                    clone_url = repo_url.replace('https://', f'https://{token}@')
                elif repo_type == RepoType.GITLAB:
                    clone_url = repo_url.replace('https://', f'https://oauth2:{token}@')
                else:
                    clone_url = repo_url
            else:
                clone_url = repo_url

            # Clone the repository
            git.Repo.clone_from(clone_url, clone_dir, **clone_options)
            return True
        except git.GitCommandError:
            return False

    try:
        # Try cloning with the specified branch
        if not attempt_clone(branch):
            # If failed, try 'master'
            if not attempt_clone('master'):
                # If 'master' also failed, try 'main'
                if not attempt_clone('main'):
                    raise Exception("Failed to clone repository with specified branch, 'master', or 'main'")

        def traverse_directory(path):
            result = {}
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                rel_path = os.path.relpath(item_path, clone_dir)

                if should_ignore(rel_path):
                    print(f"Ignoring: {rel_path}")  # Debug print
                    continue

                if os.path.isdir(item_path):
                    sub_result = traverse_directory(item_path)
                    if sub_result:  # Only include non-empty directories
                        result[item] = sub_result
                else:
                    if is_important_file(item, file_patterns):
                        try:
                            with open(item_path, 'r', encoding='utf-8') as file:
                                content = file.read()
                            result[item] = content
                        except Exception as e:
                            result[item] = f"Error reading file: {str(e)}"
                    else:
                        result[item] = "file"
            return result

        # Traverse the cloned repository
        structure = traverse_directory(clone_dir)

        return {"structure": structure}

    except git.GitCommandError as e:
        raise Exception(f"Git error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error traversing repository: {str(e)}")
    finally:
        # Clean up: remove the cloned repository
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)

@app.function(image=image, secrets=[Secret.from_name("git-traverser-secret")])
@web_endpoint(method="POST")
def get_git_structure(
    request: GitRepoRequest,
    authorization: str = Header(None),
    x_git_token: Optional[str] = Header(None, alias="X-Git-Token")
):
    try:
        # Validate bearer token
        valid_token = os.environ["API_KEY"]
        if not authorization or not validate_bearer_token(authorization, valid_token):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

        # Detect or use provided repo type
        repo_type = request.type or detect_repo_type(request.repo_url)

        structure = traverse_git_repo.remote(
            request.repo_url,
            request.branch,
            repo_type,
            x_git_token,
            request.file_patterns
        )
        return JSONResponse(content=structure)
    except HTTPException as he:
        return JSONResponse(content={"error": he.detail}, status_code=he.status_code)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    app.serve()
