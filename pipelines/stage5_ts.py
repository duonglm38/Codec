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
from pipelines.process_en_data import InputExample, save_pickle
from utils import post_process
import logging
from transformers import AutoTokenizer


log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger = logging.getLogger()


ENTITIES = ['ORG', 'PER', 'LOC']


def remap_org_data(predict_labels, tokens, org_tokens):
    if len(tokens) == org_tokens:
        return predict_labels
    p = 0
    org_p = 0
    remap_predict_labels = []
    while org_p < len(org_tokens):
        if p >= len(tokens):
            remap_predict_labels.append('O')
        elif tokens[p] == org_tokens[org_p]:
            remap_predict_labels.append(predict_labels[p])
            p += 1
            # org_p += 1
        else:
            assert org_tokens[org_p] == '[' or org_tokens[org_p] == ']' or org_tokens[org_p] == '<unk>', \
                f"\n{tokens}\n{org_tokens}"
            remap_predict_labels.append('O')
            # org_p += 1
        org_p += 1
    is_opening = False
    prev_tag = "O"
    i = 0
    while i < len(remap_predict_labels):
        tag = remap_predict_labels[i]
        if tag.startswith('B-'):
            is_opening = True
            prev_tag = tag[2:]
        elif tag == 'O':
            is_opening = False
        elif tag.startswith('I-') and not is_opening:
            j = i - 1
            while j >= 0 and remap_predict_labels[j] == 'O':
                remap_predict_labels[j] = 'I-' + prev_tag
                j -= 1
        i += 1
    return remap_predict_labels


def remap_org_vi_data(predict_labels, tokens, org_tokens):
    if len(tokens) == org_tokens:
        return predict_labels
    p = 0
    org_p = 0
    remap_predict_labels = []
    while org_p < len(org_tokens):
        if p >= len(tokens):
            remap_predict_labels.append('O')
        elif tokens[p] == org_tokens[org_p]:
            remap_predict_labels.append(predict_labels[p])
            p += 1
        else:
            org_sub_tokens = [_ for _ in org_tokens[org_p].split('_') if _ != '']
            if len(org_sub_tokens) == 1:
                assert org_tokens[org_p] == '[' or org_tokens[org_p] == ']' or org_tokens[org_p] == '<unk>', \
                    f"\n{tokens}\n{org_tokens}"
                remap_predict_labels.append('O')
            else:
                tag = None
                for sub_token in org_sub_tokens:
                    assert sub_token == tokens[p], f"{sub_token}, {tokens[p]} \n {org_tokens} \n {tokens}"
                    if tag is None and predict_labels[p] != 'O':
                        tag = predict_labels[p]
                    p += 1
                if tag is None:
                    tag = 'O'
                remap_predict_labels.append(tag)
        org_p += 1
    return remap_predict_labels


def assign_labels(candidates, template_text, org_template_tokens, tag_list, scores, lang=None):
    if lang == 'zho':
        template_tokens = zho_tokenize(template_text)
    else:
        template_tokens = template_text.split(' ')
    white_space_positions = [0]
    token_ids = []
    for idx, token in enumerate(template_tokens):
        white_space_positions.append(white_space_positions[-1] + len(template_tokens[idx]))
        token_ids.append(idx)
    predict_labels = ["O"]*len(template_tokens)
    formatted_text = []
    for idx, candidate in enumerate(candidates):
        if scores[idx] == -float('inf'):
            continue
        tag = tag_list[idx]
        candidate = post_process(candidate, template_text)
        candidate = re.sub(r'\[', r' [ ', candidate)
        candidate = re.sub(r'\]', r' ] ', candidate)
        candidate = re.sub(' +', ' ', candidate)
        candidate = candidate.strip()
        if lang == 'zho':
            candidate_tokens = zho_tokenize(candidate)
        else:
            candidate_tokens = candidate.split(" ")

        if lang == 'zho':
            formatted_candidate = candidate_tokens
        else:
            c_pointer = 0
            t_pointer = 0
            formatted_candidate = []
            while t_pointer < len(template_tokens):
                if template_tokens[t_pointer] == candidate_tokens[c_pointer]:
                    formatted_candidate.append(template_tokens[t_pointer])
                    t_pointer += 1
                    c_pointer += 1
                else:
                    if not (candidate_tokens[c_pointer] == '[' or candidate_tokens[c_pointer] == ']' or
                            candidate_tokens[c_pointer + 1] == '[' or candidate_tokens[c_pointer + 1] == ']'):
                        print(candidate_tokens)
                        exit(0)
                    if candidate_tokens[c_pointer] == ']' or candidate_tokens[c_pointer] == '[':
                        formatted_candidate.append(candidate_tokens[c_pointer])
                        c_pointer += 1
                        continue
                    elif candidate_tokens[c_pointer + 1] == '[':
                        formatted_candidate.append('[')
                        formatted_candidate.append(template_tokens[t_pointer])
                        tmp = [candidate_tokens[c_pointer]]
                        assert template_tokens[t_pointer].startswith(tmp[0])
                        c_pointer += 2
                        flag = False
                        while ''.join(tmp) != template_tokens[t_pointer]:
                            if candidate_tokens[c_pointer] != ']':
                                tmp.append(candidate_tokens[c_pointer])
                                c_pointer += 1
                            else:
                                flag = True
                                c_pointer += 1
                        if flag:
                            formatted_candidate.append(']')
                        t_pointer += 1
                    else:
                        formatted_candidate.append(template_tokens[t_pointer])
                        formatted_candidate.append(']')
                        t_pointer += 1
                        c_pointer += 3

            while c_pointer < len(candidate_tokens):
                formatted_candidate.append(candidate_tokens[c_pointer])
                c_pointer += 1
        if lang != 'zho' and not len(formatted_candidate) == len(template_tokens) + 2:
            print(formatted_candidate)
            print(template_tokens)
            exit(0)
        is_opening = False
        is_beginning = False
        for i in range(len(formatted_candidate)):
            if i < len(template_tokens) and (template_tokens[i] == '[' or template_tokens[i] == ']'):
                continue
            if formatted_candidate[i] == '[':
                is_opening = True
                is_beginning = True
                continue
            elif formatted_candidate[i] == ']':
                is_opening = False
                break
            if is_opening:
                if is_beginning:
                    predict_labels[i - 1] = f'B-{tag}'
                else:
                    predict_labels[i - 1] = f'I-{tag}'
                is_beginning = False
        if lang == 'zho':
            formatted_text.append(''.join(formatted_candidate))
        else:
            formatted_text.append(' '.join(formatted_candidate))
    if lang == 'vie':
        predict_labels = remap_org_vi_data(predict_labels=predict_labels,
                                           tokens=template_tokens,
                                           org_tokens=org_template_tokens)
    else:
        predict_labels = remap_org_data(predict_labels=predict_labels,
                                        tokens=template_tokens,
                                        org_tokens=org_template_tokens)
    return predict_labels, formatted_text


def re_ranking1(input_data, en_entity_list, lang):
    data = []
    for idx, d in enumerate(tqdm(input_data)):
        # trans_entity_list = en_examples[idx].ent_trans[args.tgt_lang]
        # trans_entity_list = [i.lower() for i in trans_entity_list]
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
            scores = []
            if len(d['tgt_lang']) == len(en_entity_list[idx]):
                flag = True
                for iidx, (r_scores, candidates) in enumerate(zip(d["reversed_scores"], d["tgt_lang"])):
                    filter_candidates = []
                    for iiidx in range(len(candidates)):
                        # candidate = candidates[iiidx]
                        marker_entity = d['tgt_entities'][iidx][iiidx]  # extract_marker_entity(candidate).lower()
                        filter_candidates.append([round(r_scores[iiidx], 4), iiidx, marker_entity])
                    best_idx = None
                    if len(filter_candidates) == 1:
                        best_idx = 0
                    elif len(filter_candidates) > 1:
                        r_score1, entity1_rank, entity1 = filter_candidates[0]
                        filter_candidates = sorted(filter_candidates)
                        _entity1 = re.sub('([?,.!:\'\"])', r' \1 ', entity1)
                        _entity1 = re.sub(' +', ' ', _entity1.strip())
                        for iiidx in range(len(filter_candidates)):
                            r_score_n, entity_n_rank, entity_n = filter_candidates[iiidx]
                            # if r_score_n < r_score1 and (entity1.endswith(" " + entity_n) or
                            #                              entity1.startswith(entity_n + " ")):
                            _entity_n = re.sub('([?,.!:\'\"])', r' \1 ', entity_n)
                            _entity_n = re.sub(' +', ' ', _entity_n.strip())
                            if r_score_n < r_score1 and (f" {_entity_n} " in f" {_entity1} " or
                                                         (lang == 'zho' and f"{_entity_n}" in f"{_entity1}")):
                                best_idx = entity_n_rank
                                break
                        if best_idx is None:
                            best_idx = entity1_rank
                    if best_idx is not None:
                        final_candidates.append(candidates[best_idx])
                        r_logp_scores.append(r_scores[best_idx])
                        scores.append(d['score'][iidx][best_idx])
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
                    "score": scores,
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


def re_ranking2(input_data, en_entity_list):
    data = []
    for idx, d in enumerate(tqdm(input_data)):
        # trans_entity_list = en_examples[idx].ent_trans[args.tgt_lang]
        # trans_entity_list = [i.lower() for i in trans_entity_list]
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
            if len(d['tgt_lang']) == len(en_entity_list[idx]):
                flag = True
                for iidx, (r_scores, candidates) in enumerate(zip(d["reversed_scores"], d["tgt_lang"])):
                    final_candidates.append(candidates[0])
                    r_logp_scores.append(r_scores[0])
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


def convert_to_hf_tokens(tokenizer, tokens, filler=False):
    token_ids = []
    for token in tokens:
        tmp = [_ for _ in tokenizer.encode(token, add_special_tokens=False) if _ != 3]
        token_ids.append(tmp)
    decoded_tokens = [tokenizer.decode(_) for _ in token_ids]
    formatted_tokens = []
    for token in decoded_tokens:
        if token != '':
            formatted_tokens.append(token)
        elif filler:
            formatted_tokens.append('<unk>')
    # formatted_tokens = [_ for _ in formatted_tokens if _ != '']
    return formatted_tokens


def zho_tokenize(text):
    tokens = [_ for _ in list(text) if _ != ' ']
    return tokens


def main(args):
    logger.info("Stage 5:......................")
    with open(args.input_file) as f:
        input_data = json.load(f)
    with open(os.path.join(args.masakhaner_path, "{}.txt".format(args.split))) as f:
        masakhaner_data = f.read().strip().split('\n\n')
    masakhaner_org_tokens = []

    for d in masakhaner_data:
        tokens = []
        lines = d.strip().split('\n')
        for line in lines:
            parts = line.strip().split(' ')
            if len(parts) != 2:
                parts = line.strip().split('\t')
            assert len(parts) == 2
            tokens.append(parts[0])
        masakhaner_org_tokens.append(tokens)
    with open(args.en_label_path) as f:
        en_examples = f.read().strip().split("\n\n")
    en_predict_entities = []

    for d in en_examples:
        entities = []
        lines = d.strip().split('\n')
        for line in lines:
            parts = line.split(' ')
            assert len(parts) == 2
            tag = parts[1]
            if tag.startswith('B-'):
                entity = tag.strip()[2:]
                entities.append(entity)
                assert entity in ENTITIES
        en_predict_entities.append(entities)

    assert len(input_data) == len(en_examples) == len(masakhaner_org_tokens)

    logger.info("Number of input examples: {}".format(len(input_data)))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # Stage 5.1: Re-ranking and filtering
    if args.mode == 1:
        data = re_ranking1(input_data=input_data,
                           en_entity_list=en_predict_entities,
                           lang=args.lang)
    else:
        data = re_ranking2(input_data=input_data,
                           en_entity_list=en_predict_entities)

    # print(json.dumps(data, indent=2))
    # Stage 5.2: create dataset
    content = []
    num_final_examples = 0
    for idx in tqdm(range(len(en_examples))):
        tag_list = en_predict_entities[idx]
        data[idx]["formatted_tgt"] = ""
        if data[idx]["flag"] == -1 or data[idx]["flag"] == 0:
            predict_labels = ["O"]*len(masakhaner_org_tokens[idx])
        else:
            if args.lang == 'zho':
                template_tokens = zho_tokenize(data[idx]["template"])
            else:
                template_tokens = data[idx]["template"].split(' ')
            formatted_template_tokens = convert_to_hf_tokens(tokenizer, template_tokens)
            if args.lang == 'zho':
                formatted_template_text = ''.join(formatted_template_tokens)
            else:
                formatted_template_text = ' '.join(formatted_template_tokens)
            formatted_org_template_tokens = convert_to_hf_tokens(tokenizer, masakhaner_org_tokens[idx], filler=True)
            assert len(formatted_org_template_tokens) == len(masakhaner_org_tokens[idx])
            predict_labels, formatted_text = assign_labels(data[idx]["tgt_lang"], formatted_template_text,
                                                           formatted_org_template_tokens, tag_list, data[idx]['score'],
                                                           args.lang)

            data[idx]["formatted_tgt"] = formatted_text
        assert len(masakhaner_org_tokens[idx]) == len(predict_labels)
        lines = ["{} {}".format(token, tag) for token, tag in zip(masakhaner_org_tokens[idx], predict_labels)]
        content.append('\n'.join(lines))

    output_file = os.path.join(args.output_path, 'stage5_mode-{}{}.txt'.format(args.mode, args.suffix))
    proj_rate = round(num_final_examples*100/len(input_data))
    logger.info("Number of output examples: {} / {}%".format(num_final_examples, proj_rate))
    output_path = os.path.join(args.output_path, 'stage5_1_mode-{}{}.json'.format(args.mode, args.suffix))
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Save to: {}".format(output_file))
    with open(output_file, 'w') as f:
        f.write('\n\n'.join(content))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--en_label_path", type=str, required=True)
    parser.add_argument("--masakhaner_path", type=str, required=True)
    parser.add_argument("--mode", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--sample_ids_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--lang", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    main(args)
