#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the runtime YAML used by the Docker container."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--bf16-target", required=True)
    parser.add_argument("--bf16-draft", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.source.open() as handle:
        config = yaml.safe_load(handle)

    config["models"]["bf16_target"] = args.bf16_target
    config["models"]["bf16_draft"] = args.bf16_draft
    config["server"]["host"] = args.host
    config["server"]["port"] = args.port

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


if __name__ == "__main__":
    main()
