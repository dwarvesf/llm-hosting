import os
import fnmatch
import shutil
from pydantic import BaseModel
from fastapi import HTTPException, Header
from fastapi.responses import JSONResponse
from modal import Image, App, web_endpoint, Secret, Volume, method, enter, exit
from typing import Optional, List
from enum import Enum
from urllib.parse import urlparse

# Create Modal Image with required dependencies
image = (
    Image.debian_slim()
    .apt_install("git")
    .pip_install("gitpython")
)

# Create Modal App
app = App(name="git-traverser")

# Create a volume to store cloned repositories
repo_volume = Volume.from_name("repo-volume", create_if_missing=True)

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
    path_parts = path.split(os.sep)
    for part in path_parts:
        if any(fnmatch.fnmatch(part, pattern) for pattern in IGNORE_PATTERNS):
            return True
    return False

def is_important_file(rel_path: str, custom_patterns: Optional[List[str]] = None) -> bool:
    patterns = custom_patterns if custom_patterns is not None else DEFAULT_IMPORTANT_FILE_PATTERNS
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in patterns)

def validate_bearer_token(bearer_token: str, valid_token: str) -> bool:
    return bearer_token == f"Bearer {valid_token}"

def detect_repo_type(repo_url: str) -> RepoType:
    if "github.com" in repo_url:
        return RepoType.GITHUB
    elif "gitlab.com" in repo_url:
        return RepoType.GITLAB
    else:
        raise ValueError("Unable to detect repository type. Please specify 'type' in the request.")

@app.cls(image=image, container_idle_timeout=30, allow_concurrent_inputs=10, volumes={"/repos": repo_volume})
class GitTraverser:
    @enter()
    def initialize(self):
        self.clone_dir = "/repos"
        if not os.path.exists(self.clone_dir):
            os.makedirs(self.clone_dir)
        repo_volume.reload()

    @exit()
    def cleanup(self):
        print("Cleaning up repository directory...")
        for item in os.listdir(self.clone_dir):
            item_path = os.path.join(self.clone_dir, item)
            if os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

        # Commit changes to the volume
        repo_volume.commit()
        print("Repository directory cleared.")

    @method()
    def traverse_git_repo(self, repo_url: str, branch: str = "main", repo_type: RepoType = None, token: Optional[str] = None, file_patterns: Optional[List[str]] = None) -> dict:
        """
        Clone a git repository blobless if it doesn't exist, traverse it, and return its directory structure.
        """

        import git

        # Detect repo type if not provided
        if repo_type is None:
            repo_type = detect_repo_type(repo_url)

        # Extract repo name from the URL
        repo_name = os.path.splitext(os.path.basename(urlparse(repo_url).path))[0]
        clone_dir = os.path.join(self.clone_dir, repo_name)

        def prepare_clone_url():
            if token:
                if repo_type == RepoType.GITHUB:
                    return repo_url.replace('https://', f'https://{token}@')
                elif repo_type == RepoType.GITLAB:
                    return repo_url.replace('https://', f'https://oauth2:{token}@')
            return repo_url

        try:
            if os.path.exists(clone_dir):
                print(f"Repository directory already exists: {clone_dir}")
                repo = git.Repo(clone_dir)
            else:
                clone_url = prepare_clone_url()
                print(f"Cloning repository: {repo_url}")
                repo = git.Repo.clone_from(clone_url, clone_dir, branch=branch, filter='blob:none')

            def traverse_directory(path='.'):
                result = {}
                try:
                    items = repo.git.ls_tree('-r', '--name-only', 'HEAD', path).splitlines()
                except git.GitCommandError as e:
                    print(f"Git error in traverse_directory: {str(e)}")
                    return result

                for item in items:
                    if should_ignore(item):
                        continue

                    parts = item.split('/')
                    current = result
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    if is_important_file(item, file_patterns):
                        try:
                            content = repo.git.show(f'HEAD:{item}')
                            current[parts[-1]] = content
                        except git.GitCommandError as e:
                            print(f"Error reading file {item}: {str(e)}")
                            current[parts[-1]] = "Error reading file"
                    else:
                        current[parts[-1]] = "file"

                return result

            # Traverse the repository
            structure = traverse_directory()

            # Commit changes to the volume
            repo_volume.commit()

            return {"structure": structure}

        except git.GitCommandError as e:
            raise Exception(f"Git command error: {str(e)}")
        except git.InvalidGitRepositoryError:
            raise Exception(f"Invalid git repository: {clone_dir}")
        except Exception as e:
            raise Exception(f"Error traversing repository: {str(e)}")

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

        structure = GitTraverser().traverse_git_repo.remote(
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
