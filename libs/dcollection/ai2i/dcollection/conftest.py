import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


@pytest.fixture(autouse=True)
def s2_api_key() -> str:
    env_secret_path = Path(__file__).parent / ".env.secret"
    load_dotenv(dotenv_path=env_secret_path, verbose=True)
    api_key = os.environ.get("S2_API_KEY")
    if not api_key:
        raise EnvironmentError(f"S2_API_KEY not found, perhaps the {env_secret_path} file is missing")
    return api_key
