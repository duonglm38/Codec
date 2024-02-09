import sys

import torch

sys.path.append('./')
import os
import re
import pickle
import json
import argparse
from tqdm import tqdm
from xml.sax.saxutils import unescape
from sacremoses import MosesTokenizer
from difflib import SequenceMatcher
from masakhaner.process_en_data import InputExample, save_pickle
from utils import post_process
import logging
from transformers import AutoTokenizer
from processing.stage5_ts import convert_to_hf_tokens


log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger = logging.getLogger()


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def load_pickle(file_name):
    # load saved results from model
    assert os.path.isfile(file_name), f"file does not exist: {file_name}"
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    print(f"Loaded data from {file_name}")
    return data


def merge_candidates(candidates, template_text, tag_list):
    open_positions = []
    close_positions = []

    template_text = re.sub(" +", " ", template_text.strip())
    try:
        candidates = [post_process(org_text=template_text, text=_) for _ in candidates]
    except:
        print(template_text)
        print(candidates)
        print()
        return None, None
    start_token_ids = [0] + [_.start() for _ in re.finditer(r' ', template_text)]

    for candidate in candidates:
        candidate = re.sub(' +', ' ', candidate)
        candidate = re.sub(r' \]', r'] ', candidate)
        candidate = re.sub(r'\[ ', r' [', candidate)
        candidate = re.sub(" +", " ", candidate.strip())

        # assert len(candidate) - 2 == len(template_text), f"{candidate}, {template_text}"
        open_position = candidate.find('[')
        if open_position - 1 >= len(template_text):
            return None, None
        if open_position > 0 and candidate[open_position - 1] == ' ' and template_text[open_position - 1] != ' ':
            candidate = re.sub(r' \[ ', r'[', candidate)
        close_position = candidate.find(']') - 1

        if close_position > open_position + 1:
            old_entity = candidate[open_position + 1:close_position]
            if not (open_position == 0 or open_position >= len(template_text) or template_text[open_position - 1] == ' '):
                if open_position > start_token_ids[-1]:
                    open_position = start_token_ids[-1] + 1
                    if start_token_ids[-1] == 0:
                        open_position = 0
                else:
                    for idx in range(1, len(start_token_ids)):
                        if open_position <= start_token_ids[idx]:
                            if idx > 1:
                                open_position = start_token_ids[idx - 1] + 1
                            else:
                                open_position = 0
                            break
            if not (open_position == 0 or open_position >= len(template_text) or template_text[open_position - 1] == ' '):
                print(template_text)
                print(candidate)
                exit(0)

            new_entity = template_text[open_position:close_position - 1]

            if not (new_entity.endswith(old_entity) and not new_entity.endswith(" " + old_entity)):
                print(template_text)
                print(candidate)
                exit(0)

        assert 0 <= open_position <= close_position
        if open_position == close_position:
            return None, None
        open_positions.append(open_position)
        close_positions.append(close_position)

    open_positions_wid = sorted([[_, i] for i, _ in enumerate(open_positions)])
    sorted_ids = [_[1] for _ in open_positions_wid]
    sorted_open_positions = [open_positions[_] for _ in sorted_ids]
    sorted_close_positions = [close_positions[_] for _ in sorted_ids]

    for i in range(len(sorted_open_positions) - 1):
        if sorted_close_positions[i] > sorted_open_positions[i + 1]:
            return None, None
    open_pointer = 0
    close_pointer = 0
    i = 0
    all_characters = []
    while i < len(template_text):
        if open_pointer < len(sorted_open_positions) and i == sorted_open_positions[open_pointer] \
                and close_pointer < len(sorted_close_positions) and i == sorted_close_positions[close_pointer]:
            all_characters.append(']')
            all_characters.append('[')
            open_pointer += 1
            close_pointer += 1
        elif open_pointer < len(sorted_open_positions) and i == sorted_open_positions[open_pointer]:
            all_characters.append('[')
            open_pointer += 1
        elif close_pointer < len(sorted_close_positions) and i == sorted_close_positions[close_pointer]:
            all_characters.append(']')
            close_pointer += 1
        else:
            all_characters.append(template_text[i])
            i += 1

    while open_pointer < len(sorted_open_positions) or close_pointer < len(sorted_close_positions):
        if open_pointer < len(sorted_open_positions) and i == sorted_open_positions[open_pointer] \
                and close_pointer < len(sorted_close_positions) and i == sorted_close_positions[close_pointer]:
            all_characters.append(']')
            all_characters.append('[')
            open_pointer += 1
            close_pointer += 1
        elif open_pointer < len(sorted_open_positions) and i == sorted_open_positions[open_pointer]:
            all_characters.append('[')
            open_pointer += 1
        elif close_pointer < len(sorted_close_positions) and i == sorted_close_positions[close_pointer]:
            all_characters.append(']')
            close_pointer += 1
        i += 1

    tag_list = [tag_list[_] for _ in sorted_ids]
    return ''.join(all_characters), tag_list


def merge_candidates_no_space(candidates, template_text, tag_list):
    open_positions = []
    close_positions = []

    template_text = template_text.strip()
    try:
        candidates = [post_process(org_text=template_text, text=_) for _ in candidates]
    except:
        print(template_text)
        print(candidates)
        print()
        return None, None

    for candidate in candidates:
        candidate = re.sub(' +', ' ', candidate)
        candidate = re.sub(r' \]', r'] ', candidate)
        candidate = re.sub(r'\[ ', r' [', candidate)
        candidate = re.sub(" +", " ", candidate.strip())

        # assert len(candidate) - 2 == len(template_text), f"{candidate}, {template_text}"
        open_position = candidate.find('[')
        if open_position - 1 >= len(template_text):
            return None, None
        if open_position > 0 and candidate[open_position - 1] == ' ' and template_text[open_position - 1] != ' ':
            candidate = re.sub(r' \[ ', r'[', candidate)
        close_position = candidate.find(']') - 1

        assert 0 <= open_position <= close_position
        if open_position == close_position:
            return None, None
        open_positions.append(open_position)
        close_positions.append(close_position)

    open_positions_wid = sorted([[_, i] for i, _ in enumerate(open_positions)])
    sorted_ids = [_[1] for _ in open_positions_wid]
    sorted_open_positions = [open_positions[_] for _ in sorted_ids]
    sorted_close_positions = [close_positions[_] for _ in sorted_ids]

    for i in range(len(sorted_open_positions) - 1):
        if sorted_close_positions[i] > sorted_open_positions[i + 1]:
            return None, None
    open_pointer = 0
    close_pointer = 0
    i = 0
    all_characters = []
    while i < len(template_text):
        if open_pointer < len(sorted_open_positions) and i == sorted_open_positions[open_pointer] \
                and close_pointer < len(sorted_close_positions) and i == sorted_close_positions[close_pointer]:
            all_characters.append(']')
            all_characters.append('[')
            open_pointer += 1
            close_pointer += 1
        elif open_pointer < len(sorted_open_positions) and i == sorted_open_positions[open_pointer]:
            all_characters.append('[')
            open_pointer += 1
        elif close_pointer < len(sorted_close_positions) and i == sorted_close_positions[close_pointer]:
            all_characters.append(']')
            close_pointer += 1
        else:
            all_characters.append(template_text[i])
            i += 1

    while open_pointer < len(sorted_open_positions) or close_pointer < len(sorted_close_positions):
        if open_pointer < len(sorted_open_positions) and i == sorted_open_positions[open_pointer] \
                and close_pointer < len(sorted_close_positions) and i == sorted_close_positions[close_pointer]:
            all_characters.append(']')
            all_characters.append('[')
            open_pointer += 1
            close_pointer += 1
        elif open_pointer < len(sorted_open_positions) and i == sorted_open_positions[open_pointer]:
            all_characters.append('[')
            open_pointer += 1
        elif close_pointer < len(sorted_close_positions) and i == sorted_close_positions[close_pointer]:
            all_characters.append(']')
            close_pointer += 1
        i += 1

    tag_list = [tag_list[_] for _ in sorted_ids]
    return ''.join(all_characters), tag_list


def extract_marker_entity(sentence):
    open_position = sentence.find('[')
    close_position = sentence.find(']')
    if close_position == open_position + 1:
        return ""
    return sentence[open_position + 1:close_position].strip()


def decode_label_span(label):
    label_tags = label
    span_labels = []
    last = 'O'
    start = -1
    for i, tag in enumerate(label_tags):
        pos, _ = (None, 'O') if tag == 'O' else tag.split('-')
        if (pos == 'B' or tag == 'O') and last != 'O':  # end of span
            span_labels.append((start, i, last.split('-')[1]))
        if pos == 'B' or last == 'O':  # start of span or move on
            start = i
        last = tag

    if label_tags[-1] != 'O':
        span_labels.append((start, len(label_tags), label_tags[-1].split('-')[1]))

    return span_labels


def marker_decode(trans_sent, tag_list, mt):
    marker_tags = ['[', ']']
    for tag in marker_tags:
        if tag in trans_sent:
            trans_sent = trans_sent.replace(tag, ' {} '.format(tag))

    sentence = trans_sent.split()

    new_sentence = []
    labels = []
    label_start = False
    label_continue = False
    lab_idx = 0

    for c in sentence:
        if c in marker_tags:
            if lab_idx >= len(tag_list):
                print(trans_sent, tag_list)
                exit(0)
            lab = tag_list[lab_idx]
            if c == '[':
                label_start = True
                label_continue = False
            elif c == ']':
                label_start = False
                lab_idx += 1
        else:
            tokens = mt.tokenize(c, escape=False)
            for idx, t in enumerate(tokens):
                t = unescape(t, {"&apos;": "'", "&quot;": '"', "&#39;": "'"})
                tokens[idx] = t

            for idx, cc in enumerate(tokens):
                if label_start:
                    if idx == 0 and label_continue is False:
                        labels.append('B-' + lab)
                        label_continue = True
                    else:
                        labels.append('I-' + lab)
                else:
                    labels.append('O')
                new_sentence.append(cc)

    if not labels:
        return None, None
    return new_sentence, labels


def marker_decode_test(tokens, labels, mt):
    new_sentence = []
    new_labels = []
    for token, label in zip(tokens, labels):
        sub_tokens = mt.tokenize(token, escape=False)
        for idx, t in enumerate(sub_tokens):
            t = unescape(t, {"&apos;": "'", "&quot;": '"', "&#39;": "'"})
            sub_tokens[idx] = t
        for idx, cc in enumerate(sub_tokens):
            tag = label
            if tag != 'O':
                tag = tag[2:]
            if tag != 'O':
                if idx == 0 and label.startswith('B-'):
                    new_labels.append(f"B-{tag}")
                else:
                    new_labels.append(f"I-{tag}")
            else:
                new_labels.append('O')
            new_sentence.append(cc)
    return new_sentence, new_labels


def re_ranking1(input_data, en_examples):
    data = []
    for idx, d in enumerate(tqdm(input_data)):
        trans_entity_list = en_examples[idx].ent_trans[args.tgt_lang]
        trans_entity_list = [i.lower() for i in trans_entity_list]
        en_entity_list = en_examples[idx].entity_list
        en_entity_list = [' '.join(tmp).lower() for tmp in en_entity_list]
        if d["flag"] == 0 or d["flag"] == -1:
            data.append({
                "template": d['template'],
                "tgt_lang": None,
                "score": d["score"],
                "reversed_scores": [],
                "flag": d["flag"]
            })
        else:
            final_candidates = []
            r_logp_scores = []
            lex_overlap_scores = []
            if len(d['tgt_lang']) == len(en_entity_list):
                flag = True
                for iidx, (r_scores, candidates) in enumerate(zip(d["reversed_scores"], d["tgt_lang"])):
                    r_score_wid = sorted([[_, i] for i, _ in enumerate(r_scores)])

                    lex_overlap_wid = []
                    for iiidx, candidate in enumerate(candidates):
                        marker_entity = extract_marker_entity(candidate).lower()
                        ratio1 = similar(marker_entity, en_entity_list[iidx])
                        ratio2 = similar(marker_entity, trans_entity_list[iidx])
                        lex_overlap_wid.append([max(ratio1, ratio2), iiidx])
                    sorted_lex_overlap_wid = sorted(lex_overlap_wid, reverse=True)

                    best_idx = None
                    if sorted_lex_overlap_wid[0][0] >= 0.9:
                        # if many items have the same ratio
                        max_ratio = sorted_lex_overlap_wid[0][0]
                        tmp_list = []
                        for ratio, org_idx in sorted_lex_overlap_wid:
                            if ratio == max_ratio:
                                tmp_list.append([r_scores[org_idx], org_idx])
                        if len(tmp_list) > 1:
                            tmp_list = sorted(tmp_list)

                        best_idx = tmp_list[0][1]
                    else:
                        for item in r_score_wid:
                            org_idx = item[1]
                            if lex_overlap_wid[org_idx][0] > 0.5:
                                best_idx = org_idx
                                break
                    if best_idx is not None:
                        final_candidates.append(candidates[best_idx])
                        r_logp_scores.append(r_scores[best_idx])
                        lex_overlap_scores.append(lex_overlap_wid[best_idx][0])
                    else:
                        flag = False
            else:
                flag = False
            if flag:
                data.append({
                    "idx": idx,
                    "src_lang": d['src_lang'],
                    "template": d['template'],
                    "tgt_lang": final_candidates,
                    "r_logp_score": r_logp_scores,
                    "lexical_score": lex_overlap_scores,
                    "flag": 1
                })
            else:
                data.append({
                    "idx": idx,
                    "src_lang": d['src_lang'],
                    "template": d['template'],
                    "tgt_lang": [None],
                    "score": [None],
                    "reversed_scores": [],
                    "flag": -1
                })
    return data


def re_ranking2(input_data, en_examples):
    data = []
    for idx, d in enumerate(tqdm(input_data)):
        trans_entity_list = en_examples[idx].ent_trans[args.tgt_lang]
        trans_entity_list = [i.lower() for i in trans_entity_list]
        en_entity_list = en_examples[idx].entity_list
        en_entity_list = [' '.join(tmp).lower() for tmp in en_entity_list]
        if d["flag"] == 0 or d["flag"] == -1:
            data.append({
                "template": d['template'],
                "tgt_lang": None,
                "score": d["score"],
                "reversed_scores": [],
                "flag": d["flag"]
            })
        else:
            final_candidates = []
            r_logp_scores = []
            lex_overlap_scores = []
            if len(d['tgt_lang']) == len(en_entity_list):
                flag = True
                for iidx, (r_scores, candidates) in enumerate(zip(d["reversed_scores"], d["tgt_lang"])):
                    best_idx = None
                    for iiidx, candidate in enumerate(candidates):
                        marker_entity = extract_marker_entity(candidate).lower()
                        ratio1 = similar(marker_entity, en_entity_list[iidx])
                        ratio2 = similar(marker_entity, trans_entity_list[iidx])
                        if max(ratio1, ratio2) > 0.5:
                            best_idx = iiidx
                            break

                    if best_idx is not None:
                        final_candidates.append(candidates[best_idx])
                        r_logp_scores.append(r_scores[best_idx])
                    else:
                        flag = False
            else:
                flag = False
            if flag:
                data.append({
                    "idx": idx,
                    "src_lang": d['src_lang'],
                    "template": d['template'],
                    "tgt_lang": final_candidates,
                    "r_logp_score": r_logp_scores,
                    "lexical_score": lex_overlap_scores,
                    "flag": 1
                })
            else:
                data.append({
                    "idx": idx,
                    "src_lang": d['src_lang'],
                    "template": d['template'],
                    "tgt_lang": [None],
                    "score": [None],
                    "reversed_scores": [],
                    "flag": -1
                })
    return data


def re_ranking3(input_data, en_examples):
    data = []
    for idx, d in enumerate(tqdm(input_data)):
        trans_entity_list = en_examples[idx].ent_trans[args.tgt_lang]
        trans_entity_list = [i.lower() for i in trans_entity_list]
        en_entity_list = en_examples[idx].entity_list
        en_entity_list = [' '.join(tmp).lower() for tmp in en_entity_list]
        if d["flag"] == 0 or d["flag"] == -1:
            data.append({
                "template": d['template'],
                "tgt_lang": None,
                "score": d["score"],
                "reversed_scores": [],
                "flag": d["flag"]
            })
        else:
            final_candidates = []
            r_logp_scores = []
            lex_overlap_scores = []
            if len(d['tgt_lang']) == len(en_entity_list):
                flag = True
                for iidx, (r_scores, candidates) in enumerate(zip(d["reversed_scores"], d["tgt_lang"])):
                    filter_candidates = []
                    for iiidx in range(len(candidates)):
                        candidate = candidates[iiidx]
                        marker_entity = extract_marker_entity(candidate).lower()
                        ratio1 = similar(marker_entity, en_entity_list[iidx])
                        ratio2 = similar(marker_entity, trans_entity_list[iidx])
                        lex_overlap = max(ratio1, ratio2)
                        if lex_overlap > 0.5:
                            filter_candidates.append([r_scores[iiidx], iiidx, lex_overlap, marker_entity])

                    best_idx = None
                    if len(filter_candidates) == 1:
                        best_idx = filter_candidates[0][1]
                    elif len(filter_candidates) > 1:
                        filter_candidates = sorted(filter_candidates)
                        r_score1, entity1_rank, lex_overlap1, entity1 = filter_candidates[0]
                        r_score2, entity2_rank, lex_overlap2, entity2 = filter_candidates[1]
                        is_suffix1 = entity2.endswith(entity1) and not (entity2.endswith(f" {entity1}"))
                        is_suffix2 = entity1.endswith(entity2) and not (entity1.endswith(f" {entity2}"))

                        if is_suffix1 or is_suffix2:
                            best_idx = min(entity1_rank, entity2_rank)
                        else:
                            lex_overlap_wid = [[_[2], _[1], _[3], __] for __, _ in enumerate(filter_candidates)]
                            sorted_lex_overlap_wid = sorted(lex_overlap_wid, reverse=True)
                            tmp_list = []
                            max_ratio = sorted_lex_overlap_wid[0][0]

                            if max_ratio >= 0.9:
                                if len(sorted_lex_overlap_wid) > 1:
                                    entity1, r_score_rank1 = sorted_lex_overlap_wid[0][2], sorted_lex_overlap_wid[0][3]
                                    entity2, r_score_rank2 = sorted_lex_overlap_wid[1][2], sorted_lex_overlap_wid[1][3]
                                    if entity2.endswith(entity1) and not (entity2.endswith(f" {entity1}")):
                                        best_idx = min(r_score_rank1, r_score_rank2)
                                    else:
                                        for ratio, org_idx, _, _ in sorted_lex_overlap_wid:
                                            if ratio == max_ratio:
                                                tmp_list.append([r_scores[org_idx], org_idx])
                                        tmp_list = sorted(tmp_list)
                                        best_idx = tmp_list[0][1]

                                else:
                                    best_idx = sorted_lex_overlap_wid[0][1]
                            else:
                                best_idx = entity1_rank
                    if best_idx is not None:
                        final_candidates.append(candidates[best_idx])
                        r_logp_scores.append(r_scores[best_idx])
                    else:
                        flag = False
            else:
                flag = False
            if flag:
                data.append({
                    "idx": idx,
                    "src_lang": d['src_lang'],
                    "template": d['template'],
                    "tgt_lang": final_candidates,
                    "r_logp_score": r_logp_scores,
                    "lexical_score": lex_overlap_scores,
                    "flag": 1
                })
            else:
                data.append({
                    "idx": idx,
                    "src_lang": d['src_lang'],
                    "template": d['template'],
                    "tgt_lang": [None],
                    "score": [None],
                    "reversed_scores": [],
                    "flag": -1
                })
    return data


def re_ranking4(input_data, en_examples, no_filtering=False):
    data = []
    for idx, d in enumerate(tqdm(input_data)):
        trans_entity_list = en_examples[idx].ent_trans[args.tgt_lang]
        trans_entity_list = [i.lower() for i in trans_entity_list]
        en_entity_list = en_examples[idx].entity_list
        en_entity_list = [' '.join(tmp).lower() for tmp in en_entity_list]
        assert len(trans_entity_list) == len(en_entity_list)
        if d["flag"] == 0 or d["flag"] == -1:
            data.append({
                "template": d['template'],
                "tgt_lang": None,
                "score": d["score"],
                "reversed_scores": [],
                "flag": d["flag"]
            })
        else:
            final_candidates = []
            r_logp_scores = []
            lex_overlap_scores = []
            if len(d['tgt_lang']) == len(en_entity_list):
                flag = True
                for iidx, (r_scores, candidates) in enumerate(zip(d["reversed_scores"], d["tgt_lang"])):
                    filter_candidates = []
                    for iiidx in range(len(candidates)):
                        candidate = candidates[iiidx]
                        marker_entity = extract_marker_entity(candidate).lower()
                        ratio1 = similar(marker_entity, en_entity_list[iidx])
                        ratio2 = similar(marker_entity, trans_entity_list[iidx])
                        lex_overlap = max(ratio1, ratio2)

                        if no_filtering or lex_overlap > 0.5 or r_scores[iiidx] < 5.:
                            filter_candidates.append([round(r_scores[iiidx], 4), iiidx, lex_overlap, marker_entity])

                    best_idx = None
                    if len(filter_candidates) == 1:
                        best_idx = filter_candidates[0][1]
                    elif len(filter_candidates) > 1:
                        lex_overlap_wid = [[-_[2], _[1], _[3], __] for __, _ in enumerate(filter_candidates)]
                        sorted_lex_overlap_wid = sorted(lex_overlap_wid)
                        tmp_list = []
                        max_ratio = -sorted_lex_overlap_wid[0][0]

                        if max_ratio >= 0.9:
                            if len(sorted_lex_overlap_wid) > 1:
                                for ratio, org_idx, _, _ in sorted_lex_overlap_wid:
                                    if -ratio == max_ratio:
                                        tmp_list.append([round(r_scores[org_idx], 4), org_idx])
                                tmp_list = sorted(tmp_list)
                                best_idx = tmp_list[0][1]
                            else:
                                best_idx = sorted_lex_overlap_wid[0][1]
                        else:
                            r_score1, entity1_rank, lex_overlap1, entity1 = filter_candidates[0]
                            filter_candidates = sorted(filter_candidates)
                            _entity1 = re.sub('([?,.!:\'\"])', r' \1 ', entity1)
                            _entity1 = re.sub(' +', ' ', _entity1.strip())
                            for iiidx in range(len(filter_candidates)):
                                r_score_n, entity_n_rank, _, entity_n = filter_candidates[iiidx]
                                # if r_score_n < r_score1 and (entity1.endswith(" " + entity_n) or
                                #                              entity1.startswith(entity_n + " ")):
                                _entity_n = re.sub('([?,.!:\'\"])', r' \1 ', entity_n)
                                _entity_n = re.sub(' +', ' ', _entity_n.strip())
                                if r_score_n < r_score1 and f" {_entity_n} " in f" {_entity1} ":
                                    best_idx = entity_n_rank
                                    break
                            if best_idx is None:
                                best_idx = entity1_rank
                    if best_idx is not None:
                        final_candidates.append(candidates[best_idx])
                        r_logp_scores.append(r_scores[best_idx])
                    else:
                        flag = False
            else:
                flag = False
            if flag:
                data.append({
                    "idx": idx,
                    "src_lang": d['src_lang'],
                    "template": d['template'],
                    "tgt_lang": final_candidates,
                    "r_logp_score": r_logp_scores,
                    "lexical_score": lex_overlap_scores,
                    "flag": 1
                })
            else:
                data.append({
                    "idx": idx,
                    "src_lang": d['src_lang'],
                    "template": d['template'],
                    "tgt_lang": [None],
                    "score": [None],
                    "reversed_scores": [],
                    "flag": -1
                })
    return data


def re_ranking4_test(input_data, en_examples, no_filtering=False):
    data = []
    for idx, d in enumerate(tqdm(input_data)):
        trans_entity_list = en_examples[idx].ent_trans[args.tgt_lang]
        trans_entity_list = [i.lower() for i in trans_entity_list]
        en_entity_list = en_examples[idx].entity_list
        en_entity_list = [' '.join(tmp).lower() for tmp in en_entity_list]
        assert len(trans_entity_list) == len(en_entity_list)
        if d["flag"] == 0 or d["flag"] == -1:
            data.append(None)
        else:
            final_candidates = []
            if len(d['tgt_lang']) == len(en_entity_list):
                flag = True
                for iidx, (r_scores, candidates) in enumerate(zip(d["reversed_scores"], d["tgt_lang"])):
                    filter_candidates = []
                    for iiidx in range(len(candidates)):
                        candidate = candidates[iiidx]
                        marker_entity = extract_marker_entity(candidate).lower()
                        ratio1 = similar(marker_entity, en_entity_list[iidx])
                        ratio2 = similar(marker_entity, trans_entity_list[iidx])
                        lex_overlap = max(ratio1, ratio2)

                        if no_filtering or lex_overlap > 0.5 or r_scores[iiidx] < 5.:
                            filter_candidates.append([round(r_scores[iiidx], 4), iiidx, lex_overlap, marker_entity])

                    best_idx = None
                    if len(filter_candidates) == 1:
                        best_idx = filter_candidates[0][1]
                    elif len(filter_candidates) > 1:
                        lex_overlap_wid = [[-_[2], _[1], _[3], __] for __, _ in enumerate(filter_candidates)]
                        sorted_lex_overlap_wid = sorted(lex_overlap_wid)
                        tmp_list = []
                        max_ratio = -sorted_lex_overlap_wid[0][0]

                        if max_ratio >= 0.9:
                            if len(sorted_lex_overlap_wid) > 1:
                                for ratio, org_idx, _, _ in sorted_lex_overlap_wid:
                                    if -ratio == max_ratio:
                                        tmp_list.append([round(r_scores[org_idx], 4), org_idx])
                                tmp_list = sorted(tmp_list)
                                best_idx = tmp_list[0][1]
                            else:
                                best_idx = sorted_lex_overlap_wid[0][1]
                        else:
                            r_score1, entity1_rank, lex_overlap1, entity1 = filter_candidates[0]
                            filter_candidates = sorted(filter_candidates)
                            # _entity1 = re.sub('([?,.!:\'\"])', r' \1 ', entity1)
                            # _entity1 = re.sub(' +', ' ', _entity1.strip())
                            for iiidx in range(len(filter_candidates)):
                                r_score_n, entity_n_rank, _, entity_n = filter_candidates[iiidx]
                                if r_score_n < r_score1 and (entity1.endswith(" " + entity_n) or
                                                             entity1.startswith(entity_n + " ")):
                                    # _entity_n = re.sub('([?,.!:\'\"])', r' \1 ', entity_n)
                                    # _entity_n = re.sub(' +', ' ', _entity_n.strip())
                                    # if r_score_n < r_score1 and f" {_entity_n} " in f" {_entity1} ":
                                    best_idx = entity_n_rank
                                    break
                            if best_idx is None:
                                best_idx = entity1_rank
                    if best_idx is not None:
                        final_candidates.append(best_idx)
                    else:
                        flag = False
            else:
                flag = False
            if flag:
                data.append(final_candidates)
            else:
                data.append(None)
    return data


def main(args):
    logger.info("Stage 5:......................")
    with open(args.input_file) as f:
        input_data = json.load(f)
    mt = MosesTokenizer(lang=args.moses_tgt_lang)
    en_examples = load_pickle(args.conll_en_pkl_path)

    if args.sample_ids_path:
        with open(args.sample_ids_path) as f:
            sample_ids = f.read().strip().splitlines()
        sample_ids = [int(_) for _ in sample_ids]

    tokenizer = None
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    with open(args.entity_trans_path) as f:
        ent_trans = f.read().strip().splitlines()
    ent_trans = [i.strip() for i in ent_trans]

    start = 0
    if args.use_partial_entity_file:
        assert args.sample_ids_path
        num_entities = 0
        for idx in sample_ids:
            tag_list = en_examples[idx].tag_list
            entity_list = en_examples[idx].entity_list
            non_misc_entity = [_ for _ in range(len(tag_list)) if tag_list[_] != 'MISC']
            tag_list = [tag_list[_] for _ in non_misc_entity]
            entity_list = [entity_list[_] for _ in non_misc_entity]
            length_entity = len(entity_list)
            end = start + length_entity
            ent_trans_tmp = ent_trans[start: end]
            assert len(ent_trans_tmp) == length_entity
            en_examples[idx].tag_list = tag_list
            en_examples[idx].entity_list = entity_list
            en_examples[idx].add_ent_translation(args.tgt_lang, ent_trans_tmp)
            start = end
            num_entities += length_entity
        assert len(ent_trans) == num_entities
    else:
        for idx, examples in enumerate(en_examples):
            length_entity = len(en_examples[idx].entity_list)
            end = start + length_entity
            ent_trans_tmp = ent_trans[start: end]
            assert len(ent_trans_tmp) == length_entity

            en_examples[idx].add_ent_translation(args.tgt_lang, ent_trans_tmp)
            start = end

    # for idx in range(len(en_examples)):
    #     tag_list = en_examples[idx].tag_list
    #     entity_list = en_examples[idx].entity_list
    #     ent_trans_list = en_examples[idx].ent_trans[args.tgt_lang]
    #
    #     non_misc_entity = [_ for _ in range(len(tag_list)) if tag_list[_] != 'MISC']
    #     tag_list = [tag_list[_] for _ in non_misc_entity]
    #     entity_list = [entity_list[_] for _ in non_misc_entity]
    #     ent_trans_list = [ent_trans_list[_] for _ in non_misc_entity]
    #     en_examples[idx].tag_list = tag_list
    #     en_examples[idx].entity_list = entity_list
    #     en_examples[idx].ent_trans[args.tgt_lang] = ent_trans_list

    if args.sample_ids_path:
        en_examples = [en_examples[_] for _ in sample_ids]
    assert len(input_data) == len(en_examples)
    logger.info("Number of input examples: {}".format(len(input_data)))
    # Stage 5.1: Re-ranking and filtering
    if args.mode == 1:
        data = re_ranking1(input_data=input_data,
                           en_examples=en_examples)
    elif args.mode == 2:
        data = re_ranking2(input_data=input_data,
                           en_examples=en_examples)
    elif args.mode == 3:
        data = re_ranking3(input_data=input_data,
                           en_examples=en_examples)
    else:
        data = re_ranking4(input_data=input_data,
                           en_examples=en_examples)

    # print(json.dumps(data, indent=2))
    # Stage 5.2: create dataset
    content = []
    num_final_examples = 0
    for idx in tqdm(range(len(en_examples))):
        tag_list = en_examples[idx].tag_list
        data[idx]["merge_tgt"] = ""
        # if '[' in data[idx]["template"] or ']' in data[idx]["template"] or len(data[idx]["template"].strip()) == 0:
        #     continue
        if len(data[idx]["template"].strip()) == 0:
            continue
        if data[idx]["flag"] == -1:
            continue
        if data[idx]["flag"] == 1:
            template_text = data[idx]["template"]
            if tokenizer:
                template_tokens = convert_to_hf_tokens(tokenizer, data[idx]["template"].split(' '))
                template_text = ' '.join(template_tokens)

            assert len(tag_list) == len(data[idx]["tgt_lang"]), f"\n{tag_list}\n{len(data[idx]['tgt_lang'])}"
            if not args.tgt_lang.startswith('zho'):
                merge_marker_text, tag_list = merge_candidates(data[idx]["tgt_lang"], template_text, tag_list)
            else:
                merge_marker_text, tag_list = merge_candidates_no_space(data[idx]["tgt_lang"], template_text, tag_list)
            data[idx]["merge_tgt"] = merge_marker_text
        else:
            merge_marker_text = data[idx]["template"]
            assert len(tag_list) == 0
        if merge_marker_text is None:
            continue
        new_sentence, labels = marker_decode(merge_marker_text, tag_list, mt)

        for word, lab in zip(new_sentence, labels):
            content.append('{} {}\n'.format(word, lab))
        content.append('\n')
        num_final_examples += 1
    output_file = os.path.join(args.output_path, 'stage5_mode-{}{}.txt'.format(args.mode, args.suffix))
    proj_rate = round(num_final_examples*100/len(input_data))
    logger.info("Number of output examples: {} / {}%".format(num_final_examples, proj_rate))
    output_path = os.path.join(args.output_path, 'stage5_1_mode-{}{}.json'.format(args.mode, args.suffix))
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Save to: {}".format(output_file))
    with open(output_file, 'w') as f:
        f.write(''.join(content))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    # parser.add_argument("--search_log_path", type=str, required=True)
    parser.add_argument("--conll_en_pkl_path", type=str, required=True)
    parser.add_argument("--entity_trans_path", type=str, required=True)
    parser.add_argument("--tgt_lang", type=str, required=True)
    parser.add_argument("--moses_tgt_lang", type=str, default="en")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--sample_ids_path", type=str, default=None)
    parser.add_argument("--no_filtering", action="store_true")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--use_partial_entity_file", action="store_true")
    args = parser.parse_args()
    main(args)

