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
import torch.nn.functional as F
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
        # Implement when GRPO config is setup
        # Copy paste sft_trainer.py __init__

        model_forward = model.forward if not isinstance(model, NeuronPeftModel) else model.get_base_model().forward
        forward_params = inspect.signature(model_forward).parameters
        
        pass

    @torch.no_grad()
    def _gather_action_logprobs(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, T, V) or (T, B, V) if sequence_parallel_enabled -> normalize to (B, T, V)
        actions: (B, T) integer indices
        returns logprobs: (B, T)
        """
        # dont think this is important either
        # delete later

        if logits.dim() != 3:
            raise ValueError(f"Expected 3D logits, got shape {tuple(logits.shape)}")
        # Trainer may run with sequence_parallel_enabled; base code transposes logits after lm_head.
        # Ensure (B, T, V).
        if logits.shape[0] < logits.shape[1]:  # heuristic if (T, B, V)
            logits = logits.transpose(0, 1).contiguous()

        log_probs = F.log_softmax(logits, dim=-1)
        # Gather log π(a|s). Add last dim to actions for gather.
        chosen_logprobs = torch.gather(log_probs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        return chosen_logprobs

    def compute_grpo_loss(
        self,
        model_outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        *,
        beta: float = 1.0,                      # GRPO temperature on advantages
        kl_coef: float = 0.0,                   # set >0 to add KL to a reference policy
    ) -> torch.Tensor:
        # lowky just replace with unsloth

        """
        A generic GRPO loss:
           w_i ∝ exp(beta * A_i)  (normalized across tokens or samples)
           L_policy = - Σ_i w_i * log πθ(a_i | s_i)
        Optional regularizers:
           + kl_coef * KL(πθ || π_ref)
           - entropy_coef * H(πθ)
        You can switch to per-sequence weighting by normalizing over batch instead of all tokens.
        """
        logits = model_outputs["logits"]                         # (B, T, V) or (T, B, V)
        actions = batch["actions"]                               # (B, T)
        advantages = batch["advantages"]                         # (B, T)

        # 1) Current policy logprobs
        logprobs = self._gather_action_logprobs(logits, actions) # (B, T)

        # 2) GRPO weights ~ exp(beta * A)
        #    Numerically stable: subtract max before exp, then normalize.
        adv = advantages.detach()
        flat = (beta * adv).view(-1)
        flat = flat - torch.max(flat)
        w = torch.exp(flat)
        w = w / (w.sum() + 1e-12)                                # normalized weights, shape (B*T,)
        w = w.view_as(adv)

        policy_loss = -(w * logprobs).sum()

        # 3) Optional KL-to-reference
        kl_loss = torch.zeros((), device=logits.device)
        if kl_coef > 0.0 and "ref_logprobs" in batch:
            ref_logprobs = batch["ref_logprobs"]                 # log π_ref(a|s), shape (B, T)
            # KL(π||π_ref) on chosen actions (lite surrogate): logπ - logπ_ref
            kl_token = (logprobs - ref_logprobs)
            kl_loss = kl_token.mean() * kl_coef                  # mean over tokens

        # 4) Optional entropy bonus (maximize entropy => subtract from loss)
        ent_loss = torch.zeros((), device=logits.device)

        loss = policy_loss + kl_loss + ent_loss
        return loss

    def train_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Copy of NeuronTrainer.train_step with one change:
        - Instead of using `outputs = model(**inputs)` loss, we compute GRPO loss.
        All accumulation/division/AMP/backward logic stays identical.
        """
        manager = self.autocast_smart_context_manager()

        # Pipeline-parallel path (NxDPPModel) uses stage-local training—usually unsuitable for custom per-token losses.
        # If you use PP, you can still route through .run_train(**inputs) and add custom loss inside the model.
        if hasattr(model, "run_train"):  # NxDPPModel
            with manager:
                loss = model.run_train(**inputs)
            # If not on last PP stage, loss is zeroed by the base class convention.
            return loss

        # Standard (non-pipeline) path
        with manager:
            # Forward to get logits without letting the model apply its own CE loss.
            outputs = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                labels=None,  # ensure the model doesn't compute CE
            )
            loss = self.compute_grpo_loss(outputs, inputs)

        # Match NeuronTrainer’s scaling behavior:
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        else:
            loss = loss / self.args.gradient_accumulation_steps

        # Backward pass using NeuronAccelerator
        self.accelerator.backward(loss)
        return loss

    # def train(self, resume_from_checkpoint: str | bool | None = None):
    #     """
    #     - beta is hyperparam for KL divergence to measure how much diverge
        
    #     TODOs
    #     overwrite train_step and add variables for ref/new/old_logits
    #     figure out where `advantages` come from
    #     """
        
    #     if resume_from_checkpoint not in [False, None]:
    #         raise ValueError("`resume_from_checkpoint` is not supported by the NeuronTrainer.")

    #     args = self.args

    #     self.accelerator.free_memory()

    #     # Data loader and number of training steps
    #     train_dataloader = self.get_train_dataloader()

    #     total_train_batch_size = self.args.train_batch_size * args.gradient_accumulation_steps
    #     (
    #         num_train_epochs,
    #         _,
    #         num_examples,
    #         _,
    #         _,
    #         len_dataloader,
    #         max_steps,
    #     ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

    #     self.setup_training(train_dataloader, max_steps, num_train_epochs, num_examples, total_train_batch_size)

    #     is_distributed = isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler)
    #     for epoch in range(num_train_epochs):
    #         # We need to call set_epoch for distributed samplers to shuffle the ordering between epochs.
    #         # See: https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    #         if is_distributed:
    #             train_dataloader.sampler.set_epoch(epoch)

    #         steps_in_epoch = (
    #             len_dataloader if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
    #         )

    #         self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

    #         step = -1
    #         epoch_iterator = iter(train_dataloader)

    #         remainder = steps_in_epoch % args.gradient_accumulation_steps
    #         if remainder == 0:
    #             remainder = args.gradient_accumulation_steps
    #         update_step = -1

    #         total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
    #             remainder < args.gradient_accumulation_steps
    #         )

    #         for _ in range(total_updates):
    #             num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
    #             batch_samples, num_items_in_batch = self.get_batch_samples(
    #                 epoch_iterator,
    #                 num_batches,
    #                 device=xm.xla_device(),
    #                 prefetch_size=args.dataloader_prefetch_size,
    #             )

    #             for inputs in batch_samples:
    #                 xm.mark_step()
    #                 step += 1
    #                 do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch

    #                 if step % args.gradient_accumulation_steps == 0:
    #                     self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

    #                 loss_step = self.train_step(self.model, inputs, num_items_in_batch=num_items_in_batch)
    #                 self.running_loss += loss_step.detach()

    #                 if do_sync_step:
    #                     self.accelerator.gradient_state.sync_gradients = True
    #                     xm.mark_step()
    #                     # Gradient clipping

    #                     self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

    #                     self.optimizer.step()
    #                     self.grad_norm = self.optimizer.grad_norm

    #                     self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

    #                     # Delay optimizer scheduling until metrics are generated
    #                     if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #                         self.lr_scheduler.step()

    #                     self.optimizer.zero_grad()

    #                     self.state.global_step += 1
    #                     self.state.epoch = epoch + (step + 1) / steps_in_epoch
    #                     self.control = self.callback_handler.on_step_end(args, self.state, self.control)
    #                     xm.mark_step()
    #                 else:
    #                     self.accelerator.gradient_state.sync_gradients = False
    #                     self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

    #                 self.maybe_log_train_step_metrics()
    #                 self.maybe_save_checkpoint()

    #                 if self.control.should_epoch_stop or self.control.should_training_stop:
    #                     # PyTorch/XLA relies on the data loader to insert the mark_step for
    #                     # each step. Since we are breaking the loop early, we need to manually
    #                     # insert the mark_step here.
    #                     xm.mark_step()
    #                     break

    #             # We also need to break out of the nested loop
    #             if self.control.should_epoch_stop or self.control.should_training_stop:
    #                 xm.mark_step()
    #                 break

    #         if step < 0:
    #             logger.warning(
    #                 "There seems to be not a single sample in your epoch_iterator, stopping training at step"
    #                 f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
    #                 f" num_steps ({max_steps}) higher than the number of available samples."
    #             )
    #             self.control.should_training_stop = True

    #         self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
    #         xm.mark_step()

    #         if self.control.should_training_stop:
    #             break

    #     logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    #     self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.get("labels")

    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")

    #     # compute custom grpo loss
    #     loss, completion_length, mean_kl = grpo_compute_loss(
    #             ref_logits,
    #             new_logits,
    #             old_logits,
    #             input_ids,
    #             mask,
    #             beta,
    #             advantages,
    #             **extra_kwargs,
    #         )

    #     return (loss, outputs) if return_outputs else loss
    
    # def accumulate_chunk(
    #         new_hidden_states_j,
    #         old_hidden_states_j,
    #         ref_hidden_states_j,
    #         input_ids_j,
    #         mask_j,
    #         advantages_j,
    #         scaling,
    #         grad_inputs_j,
    #     ):
    #         (chunk_grad_input,), (chunk_loss, (unscaled_loss, chunk_completion_length, chunk_mean_kl,)) = torch.func.grad_and_value(
    #             compute_loss,
    #             argnums = (0,),
    #             has_aux = True,
    #         )(new_hidden_states_j, old_hidden_states_j, ref_hidden_states_j, input_ids_j, mask_j, advantages_j, scaling)
    #         accumulated_loss             .add_(unscaled_loss)
    #         accumulated_completion_length.add_(chunk_completion_length)
    #         accumulated_mean_kl          .add_(chunk_mean_kl)
    #         grad_inputs_j[:] = chunk_grad_input
    #     pass