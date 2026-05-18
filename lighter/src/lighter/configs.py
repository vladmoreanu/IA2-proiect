import toml
import json

from typing import Dict, Mapping


def _to_config(value):
    if isinstance(value, Mapping) and not isinstance(value, Config):
        return Config(value)
    if isinstance(value, list):
        return [_to_config(v) for v in value]
    return value


def _normalize_assignment(keys, values):
    if not isinstance(keys, (list, tuple)):
        return [(keys, values)]

    if isinstance(values, (list, tuple)):
        if len(keys) != len(values):
            raise ValueError("Number of keys and values must match")
        return list(zip(keys, values))

    return [(k, values) for k in keys]


def _assign_dotted(mapping: Mapping, path: str, value):
    parts = path.split(".")
    current = mapping

    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], Mapping):
            raise KeyError(path)
        current = current[part]

    current[parts[-1]] = _to_config(value)


class Config(Dict):
    """
    Dict subclass supporting:
      - recursive Config conversion
      - multi-key assignment
      - dotted-path assignment ONLY

    Instantiated as any old dict is.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __setattr__(self, key, value):
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __setitem__(self, keys, values):
        for k, v in _normalize_assignment(keys, values):

            # dotted assignment allowed
            if isinstance(k, str) and "." in k:
                _assign_dotted(self, k, v)
                continue

            super().__setitem__(k, _to_config(v))

    def __getitem__(self, key):
        # Explicitly block dotted or tuple access
        if isinstance(key, tuple):
            raise TypeError("Multi-key access is not supported")

        if isinstance(key, str) and "." in key:
            raise KeyError("Scoped (dotted) access is not allowed")

        return super().__getitem__(key)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def update(self, *args, **kwargs):
        for mapping in args:
            if isinstance(mapping, Mapping):
                for k, v in mapping.items():
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def to_json(self, *, indent=2) -> str:
        '''
        Return JSON formatted string
        '''
        return json.dumps(self, indent=indent)

    def to_toml(self) -> str:
        '''
        Return TOML formatted string
        '''
        return toml.dumps(self)
    