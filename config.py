import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union


def get_key_recursive(
    dictionary: dict,
    *keys: str,
    sep: Optional[str] = None,
    silent: bool = False,
    fallback: Optional[Any] = None,
) -> Any:
    iter = keys if sep is None else keys[0].split(sep)
    res = dictionary
    try:
        for key in iter:
            res = res[key]
        return res
    except (KeyError, TypeError) as err:
        if silent:
            return fallback

        raise err


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c",
        "--config-file",
        type=str,
        help="Path to .toml config file",
        default="config.toml",
    )
    return parser


def read_config_from_file(config_file: Union[str, Path]) -> Dict[str, Any]:
    import tomli

    with open(config_file, mode="rb") as file:
        return tomli.load(file)


def read_config_from_cli(args: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = get_parser()
    parsed_args = parser.parse_args(args=args)
    return read_config_from_file(parsed_args.config_file)


class ConfigException(Exception):
    pass


class Config:
    def __init__(self, config: dict, sep: str = "."):
        self._config = config
        self.sep = sep

    @classmethod
    def from_file(cls, filename: Union[str, Path], sep: str = ".") -> "Config":
        config = read_config_from_file(filename)
        return cls(config, sep=sep)

    @classmethod
    def from_cli(cls, sep: str = "."):
        config = read_config_from_cli()
        return cls(config, sep=sep)

    @classmethod
    def from_json(cls, json_str: str, sep: str = "."):
        config = json.loads(json_str)
        return cls(config, sep=sep)

    def to_dict(self) -> dict:
        return self._config

    def to_json(self, **kwargs) -> str:
        return json.dumps(self._config, **kwargs)

    def __getitem__(self, item: str):
        try:
            return get_key_recursive(self._config, item, sep=self.sep, silent=False)
        except Exception:
            raise ConfigException(f"Could not get key {item} from configuration file")

    def get(self, item: str, default: Any):
        return get_key_recursive(
            self._config, item, sep=self.sep, silent=True, fallback=default
        )
