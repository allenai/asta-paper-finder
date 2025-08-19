import hashlib
import json
import logging
import os
from typing import Any

import git
from aiocache.base import BaseCache

logger = logging.getLogger(__name__)


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cache")


def _get_current_git_sha() -> str:
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        logger.warning("Not a git repository or git not found. Using fallback SHA.")
        return "no-git-repo-fallback-sha"


class FileBasedCache(BaseCache):
    def _build_key(self, key: str, namespace: str | None = None) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    async def _get(self, key: str, encoding: str = "utf-8", _conn: Any = None) -> str | None:
        logger.info("looking for cached results.")

        # Generate cache key
        current_sha = _get_current_git_sha()
        cache_dir = os.path.join(CACHE_DIR, current_sha)
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{key}.json")

        # Check if cache file exists
        if not os.path.exists(cache_file):
            logger.info(f"Cache miss for key: {key}")
            return None

        # Read cache file
        try:
            with open(cache_file, "r") as f:
                cached_data = f.read()
            logger.info(f"Cache hit for key: {key}")
            return cached_data
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading cache file: {e}")
            return None

    async def _set(
        self, key: str, value: str, ttl: int | None = None, _cas_token: Any = None, _conn: Any = None
    ) -> None:
        # Generate cache key
        current_sha = _get_current_git_sha()
        cache_dir = os.path.join(CACHE_DIR, current_sha)
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{key}.json")

        # Write to cache file
        try:
            with open(cache_file, "w") as f:
                f.write(value)
            logger.info(f"Saved result to cache with key: {key}")
        except IOError as e:
            logger.error(f"Error writing to cache file: {e}")
