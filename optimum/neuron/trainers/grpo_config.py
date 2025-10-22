# SAAS
#
#
from dataclasses import dataclass

from ..utils.import_utils import is_trl_available
from .grpo_training_args import GRPONeuronTrainingArguments
from .trl_utils import TRL_VERSION


if is_trl_available():
    from trl import GRPOConfig
else:

    @dataclass
    class SFTConfig:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"You need to install the `trl=={TRL_VERSION}` library to use the `NeuronSFTConfig`.")


@dataclass
class NeuronSFTConfig(GRPONeuronTrainingArguments, GRPOConfig):
    pass