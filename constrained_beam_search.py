import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from utils import Candidate
from transformers.generation.utils import BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput
import heapq
import code


def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        template_ids: torch.LongTensor,
        candidate: Candidate,
        num_return_candidates: int,
        bracket_mapping: dict,
        bracket_stack: list,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,
):
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only
    template_pointer = torch.ones(input_ids.shape[0], dtype=torch.long)
    stack_pointer = torch.ones(input_ids.shape[0], dtype=torch.long)
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
        # cannot be generated both before and after the `nn.functional.log_softmax` operation.
        next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
        inf_token_scores = torch.full(next_token_scores.shape, -float("inf")).to(next_token_scores.get_device())
        num_tokens = next_token_scores.shape[1]
        for i in range(len(template_pointer)):
            candidate_token_ids = [template_ids[template_pointer[i]].item()]
            if stack_pointer[i] >= 0:
                candidate_token_ids.extend(bracket_mapping[bracket_stack[stack_pointer[i]]])
            for token_idx in candidate_token_ids:
                inf_token_scores[i, token_idx] = next_token_scores[i, token_idx]
            # next_token_scores[i, [ii for ii in range(num_tokens) if ii not in candidate_token_ids]] = -float("inf")
        next_token_scores = inf_token_scores
        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]
        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        new_template_pointer = template_pointer.clone()
        new_stack_pointer = stack_pointer.clone()
        for i in range(len(input_ids)):
            if input_ids[i][-1] == template_ids[template_pointer[beam_idx[i]]]:
                new_template_pointer[i] = template_pointer[beam_idx[i]] + 1
                new_stack_pointer[i] = stack_pointer[beam_idx[i]]
            elif input_ids[i][-1] in bracket_mapping['['] or input_ids[i][-1] in bracket_mapping[']']:
                assert stack_pointer[beam_idx[i]] >= 0
                new_template_pointer[i] = template_pointer[beam_idx[i]]
                new_stack_pointer[i] = stack_pointer[beam_idx[i]] - 1
        template_pointer = new_template_pointer
        stack_pointer = new_stack_pointer
        # code.interact(local=dict(globals(), **locals()))
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
    )

    open_bracket_position = [-1]*input_ids.shape[0]
    close_bracket_position = [-1]*input_ids.shape[0]
    output_sequences = sequence_outputs["sequences"]
    output_scores = sequence_outputs["sequence_scores"]
    valid_hyp_ids = []

    assert len(template_ids) <= output_sequences.shape[1]
    for i in range(len(output_sequences)):
        template_p = 0
        input_p = 0
        while input_p < len(output_sequences[i]):
            if template_p < len(template_ids) and template_ids[template_p].item() == output_sequences[i][input_p].item():
                template_p += 1
                input_p += 1
            else:
                if output_sequences[i][input_p] == pad_token_id:
                    break
                assert output_sequences[i][input_p] in bracket_mapping['['] or \
                       output_sequences[i][input_p] in bracket_mapping[']']
                if output_sequences[i][input_p] in bracket_mapping['[']:
                    open_bracket_position[i] = input_p
                else:
                    close_bracket_position[i] = input_p
                input_p += 1
        if open_bracket_position[i] > -1 and close_bracket_position[i] > -1 and output_scores[i] > -float("inf"):
            valid_hyp_ids.append(i)
    valid_hyp_ids = valid_hyp_ids[:num_return_candidates]
    for i in valid_hyp_ids:
        heapq.heappush(candidate.min_heap, (output_scores[i].item(),
                                            -i,
                                            output_sequences[i].unsqueeze(0),
                                            None,
                                            [open_bracket_position[i], close_bracket_position[i]]))
    return candidate

