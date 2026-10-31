"""Training observer protocol, distinct from the frozen launcher-event schema."""

import json

TRAINING_SENTINEL = "@@TRAINING@@"


def training_line(kind, payload):
    return TRAINING_SENTINEL + json.dumps(dict(version="1.0.0", type=kind, payload=payload), ensure_ascii=False)


def parse_training_line(line):
    if not line.startswith(TRAINING_SENTINEL):
        return None
    try:
        data = json.loads(line[len(TRAINING_SENTINEL):])
        if (data.get("version") != "1.0.0" or data.get("type") not in ("PROGRESS", "REPORT", "ERROR")
                or not isinstance(data.get("payload"), dict)):
            return None
        return data
    except (ValueError, TypeError, AttributeError):
        return None
