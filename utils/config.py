import argparse
import yaml


class Config:
    """Nested config object with dot-access and YAML-friendly helpers.

    The object recursively converts nested dictionaries into ``Config`` instances,
    so callers can use attribute access like ``cfg.model.image.vit_name`` instead
    of dictionary indexing.

    Supported helpers:
    - ``Config.from_dict(...)``: build a config tree from nested mappings.
    - ``Config.from_yaml(...)``: load YAML and optionally apply overrides.
    - ``cfg.to_dict()``: convert back to plain Python dict/list scalars.
    - ``cfg.save(...)``: persist the resolved config to YAML.
    - ``Config.merge_overrides(...)``: apply dotted CLI overrides or object merge.
    """

    def __init__(self, dictionary=None):
        if dictionary:
            if not isinstance(dictionary, dict):
                raise TypeError(f"Config expects a dict-like input, got {type(dictionary)}")
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)

    @classmethod
    def from_dict(cls, dictionary=None):
        return cls(dictionary)

    @classmethod
    def from_yaml(cls, config_path: str, overrides=None):
        try:
            with open(config_path, "r") as file:
                cfg_dict = yaml.safe_load(file) or {}
            cfg = cls.from_dict(cfg_dict)
            return cls.merge_overrides(cfg, overrides)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Config file not found: {config_path}") from exc
        except Exception as exc:
            raise Exception(f"Failed to load config: {exc}") from exc

    @staticmethod
    def _to_plain(obj):
        if isinstance(obj, Config):
            return {k: Config._to_plain(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, dict):
            return {k: Config._to_plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [Config._to_plain(v) for v in obj]
        return obj

    def to_dict(self):
        return self._to_plain(self)

    def save(self, save_path: str):
        with open(save_path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=False)

    @classmethod
    def _set_by_dotted_key(cls, cfg, dotted_key: str, raw_value: str):
        if not dotted_key:
            raise ValueError("Override key cannot be empty")

        keys = dotted_key.split('.')
        node = cfg
        for key in keys[:-1]:
            child = getattr(node, key, None)
            if child is None:
                child = cls()
                setattr(node, key, child)
            elif not isinstance(child, cls):
                raise ValueError(
                    f"Cannot assign nested key '{dotted_key}': '{key}' is not a mapping node"
                )
            node = child

        setattr(node, keys[-1], yaml.safe_load(raw_value))

    @staticmethod
    def _normalize_overrides(overrides):
        if overrides is None:
            return {}
        if isinstance(overrides, Config):
            return dict(overrides.__dict__)
        if isinstance(overrides, dict):
            return dict(overrides)
        if hasattr(overrides, "__dict__"):
            return dict(overrides.__dict__)
        raise TypeError(f"Unsupported override source type: {type(overrides)}")

    @classmethod
    def merge_overrides(cls, target, overrides, *, allow_keys=None):
        """Apply overrides to a target config-like object.

        Supported override formats:
        1) Iterable[str] of dotted KEY=VALUE pairs for CLI overrides.
        2) Config/dict/object with attributes for object-to-object merge.
        """
        if isinstance(overrides, (list, tuple)):
            if not isinstance(target, cls):
                raise TypeError("Dotted KEY=VALUE overrides require target to be Config")
            for item in overrides:
                if '=' not in item:
                    raise ValueError(
                        f"Invalid override '{item}'. Expected format KEY=VALUE, "
                        "e.g. model.alignment.geo_scorer=ot"
                    )
                key, value = item.split('=', 1)
                cls._set_by_dotted_key(target, key.strip(), value.strip())
            return target

        mapping = cls._normalize_overrides(overrides)
        allow_set = set(allow_keys) if allow_keys is not None else None
        for key, value in mapping.items():
            if allow_set is not None and key not in allow_set:
                continue
            if not hasattr(target, key):
                continue
            setattr(target, key, value)
        return target

    def apply_overrides(self, overrides, *, allow_keys=None):
        return self.merge_overrides(self, overrides, allow_keys=allow_keys)


def save_config(config_obj, save_path: str):
    """Backwards-compatible wrapper around Config.save()."""
    if isinstance(config_obj, Config):
        config_obj.save(save_path)
        return
    with open(save_path, "w") as f:
        yaml.safe_dump(config_to_dict(config_obj), f, sort_keys=False, allow_unicode=False)


def config_to_dict(obj):
    """Backwards-compatible helper that converts Config-like objects to plain dicts."""
    return Config._to_plain(obj)


def build_from_defaults(config_cls, section_cfg=None):
    """Build a config-class instance from its defaults, then apply section overrides."""
    base_obj = config_cls()
    allowed = list(base_obj.__dict__.keys())
    Config.merge_overrides(base_obj, section_cfg, allow_keys=allowed)
    refresh = getattr(base_obj, "refresh_derived_fields", None)
    if callable(refresh):
        refresh()
    return base_obj


def load_config(config_path: str = "config.yaml", overrides=None):
    """Load YAML config and optionally apply dotted KEY=VALUE overrides."""
    return Config.from_yaml(config_path, overrides=overrides)


def arg_parser():
    parser = argparse.ArgumentParser("GeoAligner DDP Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode in training script",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override config value with dotted key path. "
            "Repeatable, e.g. --set model.alignment.geo_scorer=ot"
        ),
    )
    return parser