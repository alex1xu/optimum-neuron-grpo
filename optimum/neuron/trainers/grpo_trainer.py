from typing import Any
import torch
import torch_xla.core.xla_model as xm
from optimum.utils import logging
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
    GenerationConfig,
)
from neuronx_distributed.pipeline import NxDPPModel
from contextlib import nullcontext

from ..utils import is_trl_available
from .grpo_config import NeuronGRPOConfig
from .transformers import NeuronTrainer
from .trl_utils import TRL_VERSION


logger = logging.get_logger()


if is_trl_available():
    from trl import GRPOConfig, GRPOTrainer
    from trl.data_utils import is_conversational
    from trl.trainer.utils import (
        pad,
        split_tensor_dict,
        shuffle_sequence_dict,
    )
else:
    class GRPOTrainer:
        pass
    class GRPOConfig:
        pass


def identity(x):
    """Identity collator for GRPO."""
    return x


class NeuronGRPOTrainer(NeuronTrainer):
    """
    GRPO Trainer for Neuron/Trainium devices.
    
    This implementation uses a hybrid approach:
    - Generation happens on CPU/CUDA using a lightweight model
    - Training happens on Neuron/Trainium for optimal performance
    
    This is necessary because Neuron models wrapped for distributed training
    don't support efficient generation APIs.
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
        tokenizer: PreTrainedTokenizerBase | None = None,
        reward_funcs = None,
        **kwargs,
    ):
        if not is_trl_available(required_version=TRL_VERSION):
            raise RuntimeError(f"Using NeuronGRPOTrainer requires trl=={TRL_VERSION}.")

        args_is_none = args is None
        if args is None:
            args = NeuronGRPOConfig(output_dir="tmp_trainer")
        elif args.__class__.__name__ == "NeuronTrainingArguments":
            args_as_dict = args.to_dict()
            args_as_dict.update({k: getattr(args, k) for k in args_as_dict.keys() if k.endswith("_token")})
            args = NeuronGRPOConfig(**args_as_dict)

        if args_is_none:
            log_level = args.get_process_log_level()
            logging.set_verbosity(log_level)
            logging.warning(f"No `GRPOConfig` passed, using `output_dir={args.output_dir}`.")

        if data_collator is None:
            data_collator = identity

        # Store model_id before parent init
        if isinstance(model, str):
            self.model_name_or_path = model
        else:
            self.model_name_or_path = getattr(model.config, "_name_or_path", None)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class if tokenizer is None else tokenizer,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
        )

        # GRPO-specific attributes
        self.max_prompt_length = getattr(self.args, "max_prompt_length", None)
        self.max_completion_length = getattr(self.args, "max_completion_length", None)
        self.num_generations = getattr(self.args, "num_generations", 1)
        self.temperature = getattr(self.args, "temperature", 1.0)
        self.top_p = getattr(self.args, "top_p", 1.0)
        self.top_k = getattr(self.args, "top_k", None)
        self.min_p = getattr(self.args, "min_p", None)
        self.repetition_penalty = getattr(self.args, "repetition_penalty", None)
        self.chat_template_kwargs = getattr(self.args, "chat_template_kwargs", {}) or {}
        
        self.loss_type = getattr(self.args, "loss_type", "grpo")
        self.scale_rewards = getattr(self.args, "scale_rewards", "group")
        self.importance_sampling_level = getattr(self.args, "importance_sampling_level", "token")
        self.mask_truncated_completions = getattr(self.args, "mask_truncated_completions", False)
        self.beta = getattr(self.args, "beta", 0.0)
        self.epsilon_low = getattr(self.args, "epsilon", 0.1)
        self.epsilon_high = getattr(self.args, "epsilon_high", self.epsilon_low)
        
        self._step = 0
        self._buffered_inputs = None
        self._generation_step_counter = 0  # Track when to regenerate
        
        self.num_iterations = getattr(self.args, "num_iterations", 1)
        self.shuffle_dataset = getattr(self.args, "shuffle_dataset", True)
        
        # Calculate how many steps we generate for at once
        if not hasattr(self.args, "steps_per_generation"):
            # Default: generate once per gradient accumulation cycle
            self.args.steps_per_generation = self.args.gradient_accumulation_steps
        
        if not hasattr(self.args, "generation_batch_size") or self.args.generation_batch_size is None:
            self.args.generation_batch_size = self.args.per_device_train_batch_size * self.args.steps_per_generation

        # Initialize CPU model for generation
        self._init_generator_model()

        # Reward functions
        if reward_funcs is not None:
            self.reward_funcs = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
            self.reward_func_names = []
            for rf in self.reward_funcs:
                if hasattr(rf, '__name__'):
                    self.reward_func_names.append(rf.__name__)
                else:
                    self.reward_func_names.append(str(rf))
        else:
            raise ValueError("reward_funcs must be provided for GRPO training")
        
        # Reward weights
        if hasattr(self.args, 'reward_weights') and self.args.reward_weights is not None:
            self.reward_weights = torch.tensor(self.args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(self.reward_funcs), dtype=torch.float32)
        
        self.reward_processing_classes = [None] * len(self.reward_funcs)
        
        # Initialize metrics tracking
        from collections import defaultdict, deque
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self._logs = {
            "images": deque(maxlen=args.per_device_train_batch_size),
            "prompt": deque(maxlen=args.per_device_train_batch_size),
            "completion": deque(maxlen=args.per_device_train_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.per_device_train_batch_size)),
            "advantages": deque(maxlen=args.per_device_train_batch_size),
        }
        
        self.pad_token = self.processing_class.pad_token
        self.pad_token_id = self.processing_class.pad_token_id
        self.eos_token_id = self.processing_class.eos_token_id

    def _init_generator_model(self):
        """Initialize a separate model for generation on CPU/CUDA."""
        if not self.model_name_or_path:
            raise ValueError(
                "Cannot initialize generator model without model name or path. "
                "Please pass a model ID string to the trainer."
            )
        
        logger.info(f"Initializing generator model on CPU from {self.model_name_or_path}")
        
        # Load a lightweight version for generation
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,  # Use fp16 for efficiency
            low_cpu_mem_usage=True,
        )
        
        # Move to CPU (or CUDA if available and args specify)
        generator_device = getattr(self.args, "generator_device", "cpu")
        if generator_device == "cuda" and torch.cuda.is_available():
            self.generator_model = self.generator_model.to("cuda")
            logger.info("Generator model loaded on CUDA")
        else:
            self.generator_model = self.generator_model.to("cpu")
            logger.info("Generator model loaded on CPU")
        
        self.generator_model.eval()
        
        # Setup generation config
        generation_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "pad_token_id": self.processing_class.pad_token_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.top_k is not None:
            generation_kwargs["top_k"] = self.top_k
        self.generation_config = GenerationConfig(**generation_kwargs)

    def _generate_single_turn(self, prompts: list, images=None):
        """Generate completions using CPU/CUDA model."""
        # Use the generator model on CPU/CUDA
        generator_device = next(self.generator_model.parameters()).device
        
        # Build generation inputs
        processor_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "padding_side": "left",
            "max_length": self.max_prompt_length,
            "truncation": True,
            "add_special_tokens": False,
        }
        
        if is_conversational({"prompt": prompts[0]}):
            generate_inputs = self.processing_class.apply_chat_template(
                conversation=prompts,
                **processor_kwargs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **self.chat_template_kwargs,
            )
        else:
            generate_inputs = self.processing_class(text=prompts, **processor_kwargs)
        
        # Move to generator device
        generate_inputs = {
            k: v.to(generator_device) if isinstance(v, torch.Tensor) else v 
            for k, v in generate_inputs.items()
        }
        
        # Generate on CPU/CUDA
        with torch.no_grad():
            prompt_completion_ids = self.generator_model.generate(
                **generate_inputs,
                generation_config=self.generation_config,
            )
        
        # Move results back to CPU for further processing
        prompt_completion_ids = prompt_completion_ids.cpu()
        prompt_ids = generate_inputs["input_ids"].cpu()
        prompt_mask = generate_inputs["attention_mask"].cpu()
        
        # Extract completions
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1)).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # Convert to lists
        prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
        completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
        
        return prompt_ids, completion_ids, None, {}

    # Import GRPO methods that don't need modification
    from trl.trainer.grpo_trainer import GRPOTrainer as _GRPO
    
    _generate = _GRPO._generate
    _generate_and_score_completions = _GRPO._generate_and_score_completions
    _calculate_rewards = _GRPO._calculate_rewards
    _compute_loss = _GRPO._compute_loss
    _get_per_token_logps_and_entropies = _GRPO._get_per_token_logps_and_entropies

    def _set_signature_columns_if_needed(self):
        """Override to set GRPO-specific signature columns."""
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images"]

    def get_batch_samples(
        self,
        epoch_iterator,
        num_batches: int,
        device: torch.device | None = None,
        prefetch_size: int | None = None,
    ) -> tuple[list[dict[str, Any]], int | torch.Tensor | None]:
        """
        Override to handle GRPO's generation and buffering logic.
        
        The key insight: we need to generate completions for multiple steps at once,
        then buffer and return one step's worth at a time.
        
        Flow:
        1. Every `steps_per_generation` steps, collect prompts and generate
        2. Split generated batch across steps
        3. Return current step's slice
        """
        # Check if we need to generate new completions
        generate_every = self.args.steps_per_generation * self.num_iterations
        if self._generation_step_counter % generate_every == 0 or self._buffered_inputs is None:
            logger.info(f"Generating completions at generation step {self._generation_step_counter}")
            
            # Collect prompts for the full generation batch
            # We need steps_per_generation * num_batches worth of data
            total_batches_needed = self.args.steps_per_generation
            raw_batches = []
            
            for _ in range(total_batches_needed):
                try:
                    batch = next(epoch_iterator)
                    raw_batches.append(batch)
                except StopIteration:
                    break
            
            if not raw_batches:
                return [], None
            
            # Flatten batches into list of samples
            raw_samples = []
            for batch in raw_batches:
                if isinstance(batch, list):
                    raw_samples.extend(batch)
                else:
                    raw_samples.append(batch)
            
            # Generate completions for all samples
            generation_batch = self._generate_and_score_completions(raw_samples)
            
            # Shuffle and split for multiple optimizer steps
            generation_batch = shuffle_sequence_dict(generation_batch)
            generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
            self._buffered_inputs = generation_batches
            
            # Reset buffer index
            self._buffer_index = 0
        
        # Get the batch for this step
        current_batch = self._buffered_inputs[self._buffer_index]
        self._buffer_index += 1
        self._generation_step_counter += 1
        
        # If we've used all buffered batches, reset for next generation
        if self._buffer_index >= len(self._buffered_inputs):
            self._buffer_index = 0
        
        # Move to XLA device
        if device is not None and device.type == "xla":
            current_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in current_batch.items()
            }
        
        # Return as a single-item list (for compatibility with NeuronTrainer)
        # The dict is already the prepared inputs for one step
        return [current_batch], None

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | torch.Tensor | None = None,
    ):
        """
        Compute GRPO loss.
        
        This method delegates to GRPO's _compute_loss which handles the
        special prompt/completion split and advantage-based loss.
        """
        if return_outputs:
            raise ValueError("return_outputs=True is not supported for GRPO")
        
        loss = self._compute_loss(model, inputs)
        return loss

    def train_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Training step that works with GRPO's prepared inputs.
        
        The inputs dict contains: prompt_ids, completion_ids, masks, advantages, etc.
        """
        manager = self.autocast_smart_context_manager()

        if isinstance(model, NxDPPModel):
            # Pipeline parallel case
            with manager:
                loss = model.run_train(**inputs)
            
            if self.pp_rank != self.pp_size - 1:
                dtype = torch.bfloat16 if self.args.bf16 else torch.float32
                loss = torch.tensor(0, dtype=dtype).to(xm.xla_device())
        else:
            # Standard case
            with manager:
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            
            # Backward pass
            self.accelerator.backward(loss)
        
        return loss

    def train(self, resume_from_checkpoint: str | bool | None = None):
        """Use NeuronTrainer's training loop."""
        return super().train(resume_from_checkpoint=resume_from_checkpoint)