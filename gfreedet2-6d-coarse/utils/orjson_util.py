import orjson as json
from pathlib import Path
import gzip


def load_json(json_path, keystoint=False):
    # return json.loads(Path(json_path).read_bytes())
    """Loads content of a JSON file.

    :param path: Path to the JSON file. If ".json.gz" extension, opens with gzip.
    :param keystoint: If True, convert JSON keys to integers.
    :return: Content of the loaded JSON file.
    """
    path = Path(json_path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            content = f.read()
    else:
        with path.open("rb") as f:
            content = f.read()

    if keystoint:
        # orjson does not support object_hook directly, so you need to implement the conversion manually
        content = json.loads(content)
        content = {int(k) if k.isdigit() else k: v for k, v in content.items()}
    else:
        content = json.loads(content)

    return content
