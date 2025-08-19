from pathlib import Path

from ai2i.di import create_module

dal_module = create_module(name="DAL")


@dal_module.provides(scope="singleton")
async def local_storage_folder() -> Path:
    path = Path(__file__).parent.parent.parent / "async-state"
    path.mkdir(parents=True, exist_ok=True)
    return path
