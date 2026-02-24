import orjson as json
from pathlib import Path
import gzip


def load_json(json_path, keystoint=False):

    path = Path(json_path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            content = f.read()
    else:
        with path.open("rb") as f:
            content = f.read()

    if keystoint:

        content = json.loads(content)
        content = {int(k) if k.isdigit() else k: v for k, v in content.items()}
    else:
        content = json.loads(content)

    return content
