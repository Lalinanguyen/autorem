"""Child-process entry point: python -I _runner.py <program.py> <result.json>.

Loads the candidate program, calls construct(), and writes its return value as
JSON. numpy arrays and scalars are converted to plain lists and numbers. Runs in
an isolated interpreter (-I) inside a temp working directory.
"""

import json
import runpy
import sys


def _to_jsonable(obj):
    if hasattr(obj, "tolist"):  # numpy arrays and numpy scalars
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def main() -> None:
    prog_path, out_path = sys.argv[1], sys.argv[2]
    namespace = runpy.run_path(prog_path, run_name="candidate")
    construct = namespace.get("construct")
    if not callable(construct):
        raise SystemExit("program does not define construct()")
    result = _to_jsonable(construct())
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, allow_nan=True)


if __name__ == "__main__":
    main()
