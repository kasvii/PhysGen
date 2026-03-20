import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union

from omegaconf import OmegaConf, DictConfig

sys.path.append("..")
import craftsman
from craftsman.utils.typing import *

# ============ Register OmegaConf Resolvers ============= #
RESOLVERS = {
    "calc_exp_lr_decay_rate": lambda factor, n: factor ** (1.0 / n),
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "idiv": lambda a, b: a // b,
    "basename": lambda p: os.path.basename(p),
    "rmspace": lambda s, sub: str(s).replace(" ", sub),
    "tuple2": lambda s: [float(s), float(s)],
    "gt0": lambda s: s > 0,
    "cmaxgt0": lambda s: C_max(s) > 0,
    "not": lambda s: not s,
    "cmaxgt0orcmaxgt0": lambda a, b: C_max(a) > 0 or C_max(b) > 0,
}

for name, func in RESOLVERS.items():
    try:
        OmegaConf.register_new_resolver(name, func)
    except ValueError:
        pass
# ======================================================= #

def C_max(value: Any) -> float:
    if isinstance(value, (int, float)):
        return value

    value = config_to_primitive(value)
    if not isinstance(value, list):
        raise TypeError("Scalar specification only supports list, got", type(value))

    if len(value) >= 6:
        max_value = value[2]
        for i in range(4, len(value), 2):
            max_value = max(max_value, value[i])
        value = [value[0], value[1], max_value, value[3]]

    if len(value) == 3:
        value = [0] + value

    assert len(value) == 4
    _, start_value, end_value, _ = value
    return max(start_value, end_value)

@dataclass
class ExperimentConfig:
    name: str = "default"
    description: str = ""
    tag: str = ""
    seed: int = 0
    use_timestamp: bool = True
    timestamp: Optional[str] = None
    exp_root_dir: str = "outputs"

    # Auto-filled fields
    exp_dir: str = "outputs/default"
    trial_name: str = "exp"
    trial_dir: str = "outputs/default/exp"
    n_gpus: int = 1

    resume: Optional[str] = None
    data_type: str = ""
    data: dict = field(default_factory=dict)
    system_type: str = ""
    system: dict = field(default_factory=dict)
    trainer: dict = field(default_factory=dict)
    checkpoint: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.tag and not self.use_timestamp:
            raise ValueError("Either tag is specified or use_timestamp is True.")

        self.trial_name = self.tag
        if self.timestamp is None:
            self.timestamp = ""
            if self.use_timestamp:
                if self.n_gpus > 1:
                    craftsman.warn("Timestamp is disabled when using multiple GPUs. Please use a unique tag.")
                else:
                    self.timestamp = datetime.now().strftime("@%Y%m%d-%H%M%S")
        self.trial_name += self.timestamp
        self.exp_dir = os.path.join(self.exp_root_dir, self.name)
        self.trial_dir = os.path.join(self.exp_dir, self.trial_name)
        os.makedirs(self.trial_dir, exist_ok=True)


def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> Any:
    yaml_confs = [OmegaConf.create(s) if from_string else OmegaConf.load(s) for s in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return parse_structured(ExperimentConfig, cfg)


def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    return OmegaConf.structured(fields(**cfg))