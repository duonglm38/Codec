# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from constrained_decoding import bracket_constraint_decode
from constrained_decoding2 import bracket_constraint_decode as slow_bracket_constraint_decode
from constrained_beam_search import beam_search

from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import ModelOutput, logging
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput, \
    SampleEncoderDecoderOutput, SampleDecoderOnlyOutput, BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput, \
    BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput, ContrastiveSearchEncoderDecoderOutput, \
    ContrastiveSearchDecoderOnlyOutput, GenerationMixin

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)

GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]
ContrastiveSearchOutput = Union[ContrastiveSearchEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput]
GenerateOutput = Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, ContrastiveSearchOutput]


@torch.no_grad()
def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        decoding_argument: dict = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head.

    <Tip warning={true}>

    Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    model's default generation configuration. You can override any `generation_config` by passing the corresponding
    parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    For an overview of generation strategies and code examples, check out the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, the default will be used, which had the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            Custom stopping criteria that complement the default stopping criteria built from arguments and a
            generation config. If a stopping criteria is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        synced_gpus (`bool`, *optional*):
            Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
            `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
            generating before other GPUs. Otherwise it'll be set to `False`.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        kwargs:
            Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchDecoderOnlyOutput`],
                - [`~generation.SampleDecoderOnlyOutput`],
                - [`~generation.BeamSearchDecoderOnlyOutput`],
                - [`~generation.BeamSampleDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchEncoderDecoderOutput`],
                - [`~generation.SampleEncoderDecoderOutput`],
                - [`~generation.BeamSearchEncoderDecoderOutput`],
                - [`~generation.BeamSampleEncoderDecoderOutput`]
    """

    if synced_gpus is None:
        if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False

    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()

    # priority: `generation_config` argument > `model.generation_config` (the default generation config)
    if generation_config is None:
        # legacy: users may modify the model configuration to control generation -- update the generation config
        # model attribute accordingly, if it was created from the model config
        if self.generation_config._from_model_config:
            new_generation_config = GenerationConfig.from_model_config(self.config)
            if new_generation_config != self.generation_config:
                warnings.warn(
                    "You have modified the pretrained model configuration to control generation. This is a"
                    " deprecated strategy to control generation and will be removed soon, in a future version."
                    " Please use a generation configuration file (see"
                    " https://huggingface.co/docs/transformers/main_classes/text_generation)"
                )
                self.generation_config = new_generation_config
        generation_config = self.generation_config

    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
    generation_config.validate()
    self._validate_model_kwargs(model_kwargs.copy())

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    # decoder-only models should use left-padding for generation
    if not self.config.is_encoder_decoder:
        if (
                generation_config.pad_token_id is not None
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_seq_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
            "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        if not has_default_max_length:
            logger.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

    if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
        raise ValueError(
            f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
            f" the maximum length ({generation_config.max_length})"
        )
    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 8. prepare distribution pre_processing samplers
    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    # 9. prepare stopping criteria
    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )

    if decoding_argument['search_mode'] == 2:
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            max_length=1024,
            num_beam_hyps_to_keep=generation_config.num_beams,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=True
        )

        input_ids, model_kwargs = GenerationMixin._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        candidate = decoding_argument['candidate']
        return beam_search(
            self,
            input_ids,
            beam_scorer,
            num_return_candidates=decoding_argument['n_best'],
            candidate=candidate,
            template_ids=decoding_argument['template_ids'],
            bracket_stack=decoding_argument['bracket_stack'],
            bracket_mapping=decoding_argument['bracket_mapping'],
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    return search_wrapper(model=self,
                          input_ids=input_ids,
                          decoding_argument=decoding_argument,
                          logits_processor=logits_processor,
                          stopping_criteria=stopping_criteria,
                          pad_token_id=generation_config.pad_token_id,
                          eos_token_id=generation_config.eos_token_id,
                          output_scores=generation_config.output_scores,
                          return_dict_in_generate=generation_config.return_dict_in_generate,
                          synced_gpus=synced_gpus,
                          **model_kwargs)


@torch.no_grad()
def search_wrapper(
        model,
        input_ids: torch.LongTensor,
        decoding_argument: dict,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs,
):
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else model.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.generation_config.eos_token_id

    device = input_ids.get_device()

    scores = torch.zeros(1)
    template_ids = decoding_argument['template_ids']
    template_pointer = decoding_argument['template_pointer']
    bracket_stack = decoding_argument['bracket_stack']
    stack_pointer = decoding_argument['stack_pointer']
    candidate = decoding_argument['candidate']
    bracket_mapping = decoding_argument['bracket_mapping']
    future_steps = decoding_argument['future_steps']
    search_mode = decoding_argument['search_mode']
    search_batch_size = decoding_argument['batch_size']
    save_visualization = decoding_argument['save_visualization']
    possible_opening_positions = decoding_argument['possible_opening_positions']
    possible_closing_positions = decoding_argument['possible_closing_positions']
    left_marker = decoding_argument['left_marker']
    right_marker = decoding_argument['right_marker']
    n_best = decoding_argument['n_best']
    cache_tensors = dict()

    assert model_kwargs["attention_mask"].shape[0] == 1 and \
           model_kwargs["encoder_outputs"].last_hidden_state.shape[0] == 1
    cache_tensors["attention_mask"] = model_kwargs["attention_mask"].repeat(search_batch_size, 1)
    cache_tensors["encoder_last_hidden_state"] = model_kwargs["encoder_outputs"].last_hidden_state.repeat(
        search_batch_size, 1, 1)
    if device >= 0:
        template_ids = template_ids.to(device)
        template_pointer = template_pointer.to(device)
        stack_pointer = stack_pointer.to(device)
        scores = scores.to(device)
        candidate.to_cuda(device)
    # fast decoding, high GPU mem usage
    if search_mode == 0:
        accumulate_scores = [[0]]
        bracket_constraint_decode(
            input_ids=input_ids,
            future_steps=future_steps,
            model=model,
            scores=scores,
            batch_size=search_batch_size,
            curr_new_tokens=0,
            template_ids=template_ids,
            template_pointer=template_pointer,
            bracket_stack=bracket_stack,
            bracket_mapping=bracket_mapping,
            stack_pointers=stack_pointer,
            candidate=candidate,
            cache_tensors=cache_tensors,
            save_visualization=save_visualization,
            left_marker=left_marker,
            right_marker=right_marker,
            possible_opening_positions=possible_opening_positions,
            possible_closing_positions=possible_closing_positions,
            n_best=n_best,
            model_kwargs=model_kwargs,
            logits_processor=logits_processor,
            prev_accumulated_scores=accumulate_scores,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id
        )
    # slow decoding, low GPU mem usage
    else:
        accumulate_scores = [[0]]
        slow_bracket_constraint_decode(
            input_ids=input_ids,
            future_steps=future_steps,
            model=model,
            scores=scores,
            batch_size=search_batch_size,
            curr_new_tokens=0,
            template_ids=template_ids,
            template_pointer=template_pointer,
            bracket_stack=bracket_stack,
            bracket_mapping=bracket_mapping,
            stack_pointers=stack_pointer,
            candidate=candidate,
            cache_tensors=cache_tensors,
            save_visualization=save_visualization,
            left_marker=left_marker,
            right_marker=right_marker,
            possible_opening_positions=possible_opening_positions,
            possible_closing_positions=possible_closing_positions,
            n_best=n_best,
            model_kwargs=model_kwargs,
            logits_processor=logits_processor,
            prev_accumulated_scores=accumulate_scores,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id
        )
    return candidate

