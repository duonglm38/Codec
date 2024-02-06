import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from utils import Candidate, MODEL2BRACKET_IDS, Node
import heapq
import code


def bracket_constraint_decode(
        input_ids,
        template_ids,
        model,
        scores,
        curr_new_tokens,
        batch_size: int,
        template_pointer: torch.LongTensor,
        bracket_stack: list[str],
        bracket_mapping: dict,
        stack_pointers: torch.LongTensor,
        candidate: Candidate,
        model_kwargs,
        logits_processor,
        prev_accumulated_scores: list[list],
        left_marker: str,
        right_marker: str,
        n_best: int = 5,
        cache_tensors=None,
        possible_opening_positions: set[int] = None,
        possible_closing_positions: dict = None,
        open_bracket_position: list[int] = None,
        close_bracket_position: list[int] = None,
        prev_nodes: Optional[list[Node]] = None,
        save_visualization: Optional[bool] = True,
        future_steps: int = 3,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
):
    device = input_ids.get_device()
    unfinished = torch.BoolTensor(input_ids.shape[0]).to(device).fill_(True)
    if curr_new_tokens > 0:
        for i in range(input_ids.shape[0]):
            if input_ids[i, -1] == eos_token_id:
                unfinished[i] = False

                if stack_pointers[i] == -1 and len(candidate.min_heap) < n_best:
                    heapq.heappush(candidate.min_heap, (scores[i].item(),
                                                        -candidate.count,
                                                        input_ids[i].unsqueeze(0),
                                                        prev_accumulated_scores[i],
                                                        [open_bracket_position[i], close_bracket_position[i]]))
                    candidate.count += 1
                    # candidate.update_smallest_candidate()
                elif stack_pointers[i] == -1 and scores[i].item() > candidate.score:
                    heapq.heappushpop(candidate.min_heap, (scores[i].item(),
                                                           -candidate.count,
                                                           input_ids[i].unsqueeze(0),
                                                           prev_accumulated_scores[i],
                                                           [open_bracket_position[i], close_bracket_position[i]]))
                    candidate.count += 1
                    candidate.update_smallest_candidate()

    if not torch.any(unfinished):
        return
    if not torch.all(unfinished):
        unfinished_ids = torch.nonzero(unfinished).view(-1)
        input_ids = input_ids[unfinished_ids]
        scores = scores[unfinished_ids]
        template_pointer = template_pointer[unfinished_ids]
        stack_pointers = stack_pointers[unfinished_ids]
        prev_accumulated_scores = [prev_accumulated_scores[_] for _ in unfinished_ids]
        if save_visualization:
            prev_nodes = [prev_nodes[_] for _ in unfinished_ids]
        open_bracket_position = [open_bracket_position[_] for _ in unfinished_ids]
        close_bracket_position = [close_bracket_position[_] for _ in unfinished_ids]

        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = reorder_cache(model_kwargs["past_key_values"], unfinished_ids)

    # prepare model inputs
    # {'input_ids': tensor (B x L), 'past_key_values': None, 'use_cache': None, 'position_ids': None,
    #  'attention_mask': None, 'token_type_ids': None}
    if "attention_mask" in model_kwargs:
        model_kwargs["attention_mask"] = cache_tensors["attention_mask"][:input_ids.shape[0]]
    if "encoder_outputs" in model_kwargs:
        model_kwargs['encoder_outputs'].last_hidden_state = \
            cache_tensors['encoder_last_hidden_state'][:input_ids.shape[0]]
    model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
    # if curr_new_tokens > 3:
    #     exit(0)
    # code.interact(local=dict(globals(), **locals()))
    outputs = model(
        **model_inputs,
        return_dict=True
    )

    model_kwargs = model._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder)

    # logits: BxLxV -> BxV
    next_token_logits = outputs.logits[:, -1, :]
    # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
    # cannot be generated both before and after the `nn.functional.log_softmax` operation.
    # pre-process distribution
    # BxV
    next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=input_ids.shape[1])
    next_token_scores = nn.functional.log_softmax(
        next_token_logits, dim=-1
    )

    next_token_scores = logits_processor(input_ids, next_token_scores)

    if open_bracket_position is None:
        open_bracket_position = [-1]*len(input_ids)
    if close_bracket_position is None:
        close_bracket_position = [-1]*len(input_ids)
    prev_batch_ids = []
    new_template_pointers = []
    new_stack_pointers = []
    flatten_next_token_upperbounds = []
    token_candidate_ids = []
    accumulated_scores = []
    nodes = []
    all_open_bracket_position = []
    all_close_bracket_position = []

    for i in range(next_token_scores.shape[0]):
        current_best_score = -float("inf")
        if candidate.accumulate_scores is not None:
            if future_steps < 0:
                current_best_score = candidate.accumulate_scores[-1]
            else:
                end_idx = max(input_ids.shape[1] + future_steps, candidate.close_bracket_position)
                end_idx = min(end_idx, len(candidate.accumulate_scores) - 1)
                current_best_score = candidate.accumulate_scores[end_idx]

        token_idx = template_ids[template_pointer[i]].item()
        # if not (token_idx in bracket_mapping['['] and stack_pointers[i] >= 0
        #         and bracket_stack[stack_pointers[i]] == ']'):
        if True:
            upperbound = next_token_scores[i, token_idx] + scores[i]
            node = Node(text=token_idx,
                        log_prob=next_token_scores[i, token_idx].item(),
                        acc_log_prob=upperbound.item(),
                        upperbound=current_best_score
                        )

            if upperbound > current_best_score:
                token_candidate_ids.append(token_idx)
                flatten_next_token_upperbounds.append(upperbound)
                prev_batch_ids.append(i)
                new_template_pointers.append(template_pointer[i] + 1)
                new_stack_pointers.append(stack_pointers[i])
                accumulated_score = prev_accumulated_scores[i] + [upperbound.item()]
                accumulated_scores.append(accumulated_score)
                nodes.append(node)
                all_open_bracket_position.append(open_bracket_position[i])
                all_close_bracket_position.append(close_bracket_position[i])

            if prev_nodes:
                prev_nodes[i].add_child_node(node)

        if possible_opening_positions and stack_pointers[i] >= 0 and bracket_stack[stack_pointers[i]] == left_marker and \
                template_pointer[i].item() not in possible_opening_positions:
            continue
        if stack_pointers[i] >= 0 and bracket_stack[stack_pointers[i]] == right_marker and possible_closing_positions:
            open_position = open_bracket_position[i]
            _possible_closing_positions = possible_closing_positions[open_position]
            if _possible_closing_positions and template_pointer[i].item() not in _possible_closing_positions:
                continue
        if stack_pointers[i] >= 0:
            bracket_ids = bracket_mapping[bracket_stack[stack_pointers[i]]]

            for bracket_id in bracket_ids:
                upperbound = next_token_scores[i, bracket_id] + scores[i]
                node = Node(text=bracket_id,
                            log_prob=next_token_scores[i, bracket_id].item(),
                            acc_log_prob=upperbound.item(),
                            upperbound=current_best_score)
                if prev_nodes:
                    prev_nodes[i].add_child_node(node)
                if upperbound > current_best_score:
                    token_candidate_ids.append(bracket_id)
                    flatten_next_token_upperbounds.append(upperbound)
                    prev_batch_ids.append(i)
                    new_template_pointers.append(template_pointer[i])
                    new_stack_pointers.append(stack_pointers[i] - 1)
                    accumulated_score = prev_accumulated_scores[i] + [upperbound.item()]
                    accumulated_scores.append(accumulated_score)
                    nodes.append(node)
                    if bracket_stack[stack_pointers[i]] == right_marker:
                        all_close_bracket_position.append(len(input_ids[i]))
                        all_open_bracket_position.append(open_bracket_position[i])
                    else:
                        all_close_bracket_position.append(close_bracket_position[i])
                        all_open_bracket_position.append(len(input_ids[i]))

    if len(flatten_next_token_upperbounds) == 0:
        return
    if curr_new_tokens == 0 and save_visualization:
        candidate.search_tree = nodes[0]
    prev_batch_ids = torch.LongTensor(prev_batch_ids)
    new_template_pointers = torch.stack(new_template_pointers)
    new_stack_pointers = torch.stack(new_stack_pointers)
    token_candidate_ids = torch.LongTensor(token_candidate_ids)

    flatten_next_token_upperbounds = torch.stack(flatten_next_token_upperbounds)
    if device >= 0:
        token_candidate_ids = token_candidate_ids.to(device)
        prev_batch_ids = prev_batch_ids.to(device)

    arg_sorted_filter_flatten_upperbounds = torch.argsort(flatten_next_token_upperbounds, descending=True)
    sorted_upperbounds = flatten_next_token_upperbounds[arg_sorted_filter_flatten_upperbounds]
    sorted_origin_token_ids = token_candidate_ids[arg_sorted_filter_flatten_upperbounds]
    prev_batch_ids = prev_batch_ids[arg_sorted_filter_flatten_upperbounds]
    new_template_pointers = new_template_pointers[arg_sorted_filter_flatten_upperbounds]
    new_stack_pointers = new_stack_pointers[arg_sorted_filter_flatten_upperbounds]
    accumulated_scores = [accumulated_scores[_] for _ in arg_sorted_filter_flatten_upperbounds]
    if save_visualization:
        nodes = [nodes[_] for _ in arg_sorted_filter_flatten_upperbounds]
    all_close_bracket_position = [all_close_bracket_position[_] for _ in arg_sorted_filter_flatten_upperbounds]
    all_open_bracket_position = [all_open_bracket_position[_] for _ in arg_sorted_filter_flatten_upperbounds]
    batch_size = batch_size
    idx_pointer = 0
    # iter over vocab

    flag = False
    while idx_pointer < len(sorted_origin_token_ids):
        # batch forming
        end_batch_idx = min(idx_pointer + batch_size, len(sorted_origin_token_ids))
        real_end_batch_idx = 0

        for i in range(idx_pointer, end_batch_idx):
            upperbound = sorted_upperbounds[i]
            if upperbound <= candidate.score:
                flag = True
                break
            real_end_batch_idx = i
        real_end_batch_idx += 1

        batch_prev_batch_ids = prev_batch_ids[idx_pointer:real_end_batch_idx]

        batch_prev_input_ids = input_ids[batch_prev_batch_ids]
        batch_next_token_ids = sorted_origin_token_ids[idx_pointer:real_end_batch_idx]
        batch_next_input_ids = torch.cat([batch_prev_input_ids, batch_next_token_ids.unsqueeze(1)], dim=-1)
        batch_scores = sorted_upperbounds[idx_pointer:real_end_batch_idx]
        batch_nodes = []
        if save_visualization:
            batch_nodes = nodes[idx_pointer:real_end_batch_idx]
        batch_open_bracket_position = all_open_bracket_position[idx_pointer:real_end_batch_idx]
        batch_close_bracket_position = all_close_bracket_position[idx_pointer:real_end_batch_idx]
        batch_template_pointer = new_template_pointers[idx_pointer:real_end_batch_idx]
        batch_stack_pointer = new_stack_pointers[idx_pointer:real_end_batch_idx]
        batch_accumulated_scores = accumulated_scores[idx_pointer:real_end_batch_idx]

        active_batch_size = len(batch_prev_batch_ids)
        idx_pointer = real_end_batch_idx

        if active_batch_size == 0:
            break

        copy_model_kwargs = dict()
        for k in model_kwargs:
            copy_model_kwargs[k] = model_kwargs[k]
        if copy_model_kwargs["past_key_values"] is not None:
            copy_model_kwargs["past_key_values"] = reorder_cache(copy_model_kwargs["past_key_values"],
                                                                 batch_prev_batch_ids)

        bracket_constraint_decode(input_ids=batch_next_input_ids,
                                  template_ids=template_ids,
                                  model=model,
                                  scores=batch_scores,
                                  curr_new_tokens=curr_new_tokens + 1,
                                  batch_size=batch_size,
                                  template_pointer=batch_template_pointer,
                                  bracket_stack=bracket_stack,
                                  bracket_mapping=bracket_mapping,
                                  stack_pointers=batch_stack_pointer,
                                  candidate=candidate,
                                  model_kwargs=copy_model_kwargs,
                                  prev_nodes=batch_nodes,
                                  save_visualization=save_visualization,
                                  left_marker=left_marker,
                                  right_marker=right_marker,
                                  n_best=n_best,
                                  cache_tensors=cache_tensors,
                                  possible_opening_positions=possible_opening_positions,
                                  possible_closing_positions=possible_closing_positions,
                                  open_bracket_position=batch_open_bracket_position,
                                  close_bracket_position=batch_close_bracket_position,
                                  prev_accumulated_scores=batch_accumulated_scores,
                                  future_steps=future_steps,
                                  pad_token_id=pad_token_id,
                                  eos_token_id=eos_token_id,
                                  logits_processor=logits_processor)

        if flag:
            break


def reorder_cache(past_key_values, beam_idx):
    reordered_past = ()
    for layer_past in past_key_values:
        reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
    return reordered_past
