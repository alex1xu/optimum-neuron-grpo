# coding=utf-8
"""
GRPO trainer adapted for Neuron.

This module provides a Neuron-compatible wrapper around the TRL `GRPOTrainer` in the
same spirit as `sft_trainer.py` provides `NeuronSFTTrainer`.

The file exposes `NeuronGRPOTrainer` which subclasses a dynamically created
`_GRPOTrainer` (merging `NeuronTrainer` and `GRPOTrainer`). When TRL is not
available, lightweight placeholders are provided so the import doesn't fail.
"""

from typing import Any

import torch
from optimum.utils import logging
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..utils import is_trl_available
from ..utils.import_utils import is_peft_available

from .transformers import NeuronTrainer
from .trl_utils import TRL_VERSION


logger = logging.get_logger()


if is_trl_available():
    # Import TRL classes only when available to avoid hard dependency at import time.
    from trl import GRPOConfig, GRPOTrainer  # type: ignore
else:

    class GRPOTrainer:
        """Placeholder used when `trl` is not installed."""


    class GRPOConfig:
        """Placeholder config used when `trl` is not installed."""


# Create a new class that inherits from NeuronTrainer and uses the source methods from GRPOTrainer.
# This mirrors the approach used for `NeuronSFTTrainer` in `sft_trainer.py`.
_GRPOTrainer = type(
    "_GRPOTrainer",
    (NeuronTrainer,),
    GRPOTrainer.__dict__.copy()
)


class NeuronGRPOTrainer(_GRPOTrainer):
    """
    `GRPOTrainer` adapted for Neuron.

    This trainer wraps the TRL `GRPOTrainer` implementation but uses `NeuronTrainer` as
    the backbone so that all Neuron-specific training primitives (accelerator, checkpointing,
    model saving in a distributed environment, etc.) are preserved.

    The implementation here provides a minimal, well-documented backbone. It aims to be
    a drop-in replacement for scripts that expect a TRL `GRPOTrainer`, while remaining
    lightweight so tests and imports still work when `trl` is not installed.

    The main responsibilities implemented here are:
      - Accepting the same constructor signature as `GRPOTrainer` (best-effort).
      - Delegating heavy-lifting to `NeuronTrainer.__init__` and the inherited TRL methods.
      - Exposing a `train()` method which defers to the Neuron training loop.
    """

    def __init__(
        self,
        model: PreTrainedModel | torch.nn.Module | str,
        args: GRPOConfig | None = None,
        data_collator: Any | None = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks: list | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, Any] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,  # deprecated
        peft_config: Any | None = None,
        **kwargs,
    ):
        # Ensure TRL is present and has an expected version when constructing the Neuron wrapper.
        if not is_trl_available(required_version=TRL_VERSION):
            raise RuntimeError(f"Using NeuronGRPOTrainer requires trl=={TRL_VERSION}.")

        # If no args are provided, create a default GRPOConfig so downstream code can rely on attributes.
        args_is_none = args is None
        if args is None:
            args = GRPOConfig()

        # Align deprecated tokenizer arg to processing_class to keep parity with NeuronTrainer.
        if tokenizer is not None and processing_class is None:
            processing_class = tokenizer

        # Set logging verbosity according to the args (if they expose that API).
        try:
            log_level = args.get_process_log_level()
            logging.set_verbosity(log_level)
        except Exception:
            # If the GRPOConfig doesn't expose this, ignore.
            pass

        # Call NeuronTrainer.__init__ to setup Neuron training infra. We intentionally keep
        # the call surface minimal and rely on the TRL mix-in to add GRPO-specific methods.
        NeuronTrainer.__init__(
            self,
            model,
            args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            tokenizer=tokenizer,
        )

        # If the wrapped TRL trainer provides any tagging or metadata hooks, call them.
        if hasattr(self.model, "add_model_tags"):
            try:
                self.model.add_model_tags(getattr(self, "_tag_names", None))
            except Exception:
                # Non-fatal: tagging is optional
                logger.debug("add_model_tags hook failed or not provided by the model.")

    def train(self, resume_from_checkpoint: str | bool | None = None):
        """
        Run training using the Neuron training loop.

        We delegate to `NeuronTrainer.train` which implements the XLA/Neuron training
        loop and co-ordinates callbacks, checkpointing, and the optimizer steps.
        """

        return NeuronTrainer.train(
            self,
            resume_from_checkpoint=resume_from_checkpoint
        )

    # Optionally, keep a small helper which mirrors SFTTrainer's non-packed dataloader preparation
    # if GRPO training needs dataset tokenization helpers in the future.
    def _prepare_non_packed_dataloader(self, *args, **kwargs):
        """Default passthrough to support scripts that call this method.

        The real GRPOTrainer from TRL may implement specialized data preparation; here we
        simply raise NotImplementedError to indicate scripts should fallback to other logic
        or the TRL-provided methods.
        """

        raise NotImplementedError("_prepare_non_packed_dataloader is trainer-specific and not implemented in the Neuron wrapper.")
