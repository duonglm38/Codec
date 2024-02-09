import os
import sys
sys.path.append('./')

import argparse
import logging
import re
import json
import math
from tqdm import tqdm
from sacremoses import MosesTokenizer, MosesDetokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from utils import preprocess2, enc_dec_scoring, preprocess


log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger = logging.getLogger()


def main(args):
    logger.info("Stage 1:......................")
    # load text data
    with open(args.input_path) as f:
        conll_data = f.read().strip().split('\n\n')
    org_en_text = None
    if args.org_path:
        with open(args.org_path) as f:
            org_en_text = f.read().strip().splitlines()
    with open(args.template_path) as f:
        template_text = f.read().strip().splitlines()
    with open(args.org_template_path) as f:
        org_template_text = f.read().strip().splitlines()
    if args.is_tokenized:
        processed_template_text = []
        tgt_mt, tgt_md = MosesTokenizer(lang=args.moses_tgt_lang), MosesDetokenizer(lang=args.moses_tgt_lang)
        for idx in tqdm(range(len(template_text))):
            _template_text = template_text[idx]
            _org_template_text = org_template_text[idx]
            tmp = preprocess2(text=_template_text.split(" "),
                              org_text=_org_template_text,
                              mt=tgt_mt,
                              md=tgt_md,
                              is_tokenized=True)
            processed_template_text.append(tmp)
        template_text = processed_template_text

    # Stage 1.1: split multi-marker text into single-marker texts
    # Output: data = [{
    #     "src_org_text": ...,
    #     "text_to_decode": [..., ...]
    # }]
    data = []
    num_invalid = 0
    num_blank = 0
    assert len(template_text) == len(conll_data) == len(org_template_text)
    for i, d in tqdm(enumerate(conll_data)):
        lines = d.split('\n')
        num_tokens = len(lines)
        tokens = []
        left_marker_pos = []
        right_marker_pos = []
        is_opening = False
        num_entities = 0
        for idx, line in enumerate(lines):
            parts = line.split(' ')
            assert len(parts) == 2
            tok, tag = parts
            tokens.append(tok)
            if tag.startswith('B-'):
                if is_opening:
                    right_marker_pos.append(idx)
                left_marker_pos.append(idx)
                is_opening = True
                num_entities += 1
            elif tag == 'O':
                if is_opening:
                    right_marker_pos.append(idx)
                    is_opening = False
        if is_opening:
            right_marker_pos.append(num_tokens)
        assert len(left_marker_pos) == len(right_marker_pos) == num_entities

        no_marker_text = ' '.join(tokens)
        src_org_text = re.sub(' +', ' ', no_marker_text.strip())
        if org_en_text:
            assert src_org_text.startswith(org_en_text[i]), f"\n{src_org_text}\n{org_en_text[i]}"
            src_org_text = org_en_text[i]
            tokens = org_en_text[i].split(' ')
        text_to_decode = []
        src_entities = []
        if len(left_marker_pos) == 0:
            text_to_decode.append(src_org_text)
            num_blank += 1
            flag = 0
        else:
            for idx in range(len(left_marker_pos)):
                partial_marker_text = tokens.copy()
                partial_marker_text.insert(right_marker_pos[idx], ']')
                partial_marker_text.insert(left_marker_pos[idx], '[')
                entity = ' '.join(partial_marker_text[left_marker_pos[idx] + 1:right_marker_pos[idx] + 1]).strip()
                partial_marker_text = re.sub(' +', ' ', ' '.join(partial_marker_text).strip())
                text_to_decode.append(partial_marker_text)
                src_entities.append(entity)
            flag = 1

        item = {
            "src_org_text": src_org_text,
            "src_entities": src_entities,
            "template": template_text[i],
            "text_to_decode": text_to_decode,
            "flag": flag
        }
        data.append(item)
    logger.info("Number of input examples: {}".format(len(conll_data)))
    logger.info("Number of invalid / blank examples: {} / {}".format(num_invalid, num_blank))

    # Stage 1.2: Compute transition log prob
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, max_length=args.max_length,
                                              src_lang=args.src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.to('cuda')
    template_text_ids = tokenizer(template_text, max_length=args.max_length, truncation=True,
                                  add_special_tokens=False).input_ids
    input_text = []
    org_idx = []
    for idx, d in enumerate(data):
        input_text.append(d['src_org_text'])
        input_text.extend(d['text_to_decode'])
        num_text_to_compute = 1 + len(d['text_to_decode'])
        for _ in range(num_text_to_compute):
            org_idx.append(idx)
    # start_idx = 0
    # end_idx = start_idx + args.batch_size
    all_lb_scores = []
    num_epochs = math.ceil(len(input_text) / args.batch_size)
    for start_idx in tqdm(range(0, num_epochs * args.batch_size, args.batch_size)):
        # while start_idx < len(input_text):
        end_idx = min(start_idx + args.batch_size, len(input_text))
        batch_input_text = input_text[start_idx:end_idx]
        batch_input = tokenizer(batch_input_text, return_tensors="pt", max_length=args.max_length,
                                truncation=True, padding=True)
        batch_input_ids = batch_input.input_ids
        batch_attention_mask = batch_input.attention_mask

        batch_template_ids = []
        for idx in range(start_idx, end_idx):
            mapped_idx = org_idx[idx]
            template_ids = [tokenizer.eos_token_id, tokenizer.lang_code_to_id[args.tgt_lang]] + template_text_ids[mapped_idx] + [tokenizer.eos_token_id]
            batch_template_ids.append(torch.LongTensor(template_ids))
        batch_lb_token_ids = pad_sequence(batch_template_ids, batch_first=True, padding_value=-100)
        if use_cuda:
            batch_input_ids = batch_input_ids.to('cuda')
            batch_attention_mask = batch_attention_mask.to('cuda')
            batch_lb_token_ids = batch_lb_token_ids.to('cuda')
        lb_scores = enc_dec_scoring(input_ids=batch_input_ids,
                                    target_ids=batch_lb_token_ids,
                                    attention_mask=batch_attention_mask,
                                    model=model)
        all_lb_scores.extend(lb_scores)

    # Log results
    logger.info("Finish stage 1:")
    logger.info("Number of output examples: {}".format(len(data)))
    output_json_path = os.path.join(args.output_path, "stage1.json")
    output_pt_path = os.path.join(args.output_path, "stage1.pt")
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    torch.save(all_lb_scores, output_pt_path)
    logger.info("Save data to: {}, {}".format(output_json_path, output_pt_path))
    logger.info("==========================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--org_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--template_path", type=str, required=True)
    parser.add_argument("--org_template_path", type=str, required=True)
    parser.add_argument("--moses_src_lang", type=str, default="en")
    parser.add_argument("--moses_tgt_lang", type=str, default="en")
    parser.add_argument("--src_lang", type=str, default="eng_Latn")
    parser.add_argument("--tgt_lang", type=str, required=True)
    parser.add_argument("--is_tokenized", action="store_true")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no_filtering", action="store_true")
    args = parser.parse_args()
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_name_or_path
    main(args)

