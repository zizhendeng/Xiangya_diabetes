"""Keep licensed inputs and generated outputs outside the code repository."""
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def require_input(value, directory=False):
    path = Path(value).expanduser().resolve(strict=True)
    if path == REPOSITORY or REPOSITORY in path.parents:
        raise ValueError('Store study inputs outside this code repository.')
    if directory and not path.is_dir():
        raise ValueError('The input directory does not exist.')
    if not directory and not path.is_file():
        raise ValueError('The input file does not exist.')
    return path


def prepare_output(value, inputs):
    path = Path(value).expanduser().resolve()
    if path == REPOSITORY or REPOSITORY in path.parents or path in REPOSITORY.parents:
        raise ValueError('Choose an output directory outside the code repository.')
    for source in inputs:
        source = Path(source).resolve()
        folder = source if source.is_dir() else source.parent
        if path == folder or folder in path.parents or path in folder.parents:
            raise ValueError('Use a separate output directory, not an input directory or its parent.')
    path.mkdir(parents=True, exist_ok=True)
    return path
