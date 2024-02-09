import math
import sys
sys.path.append('./')
import os
import torch
import argparse
import json
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from tqdm import tqdm
from utils import enc_dec_scoring, preprocess, post_process
import re
import logging


log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger = logging.getLogger()


def main(args):
    logger.info("Stage 4:......................")
    # Stage 4.1: re-map tokenized text to original text
    if args.merge_shards:
        f_names = [[int(_.split('.')[-2]), _] for _ in os.listdir(args.input_path) if _.endswith('.json')]
        f_names = [_[1] for _ in sorted(f_names)]
        normalized_input_data = []
        for f_name in f_names:
            with open(os.path.join(args.input_path, f_name)) as f:
                normalized_input_data.extend(json.load(f))
    else:
        with open(args.input_path) as f:
            normalized_input_data = json.load(f)
    logger.info("Number of input examples: {}".format(len(normalized_input_data)))
    # 4.2: Compute log prob
    src_text = []
    tgt_text = []
    idx_mapping = dict()

    for idx, d in enumerate(normalized_input_data):
        if d['flag'] == 0 or d['flag'] == -1:
            continue
        for iidx, partial_text in enumerate(d['src_lang']):
            # src_entity = re.search(r'\[(.+?)\]', partial_text).group(1).strip()
            src_entity = d['src_entities'][iidx]
            if len(d['src_lang']) != len(d['tgt_lang']):
                import code
                code.interact(local=dict(globals(), **locals()))
                exit(0)
            for iiidx, candidate_text in enumerate(d['tgt_lang'][iidx]):
                # tgt_entity = re.search(r'\[(.+?)\]', candidate_text)
                # assert tgt_entity is not None
                # tgt_entity = tgt_entity.group(1).strip()
                tgt_entity = d['tgt_entities'][iidx][iiidx]
                if args.src_lang.startswith('zh'):
                    tgt_entity = tgt_entity.replace(' ', '')
                # d['tgt_entities'][iidx][iiidx] = tgt_entity
                idx_mapping[(idx, iidx, iiidx)] = len(tgt_text)
                src_text.append(src_entity)
                tgt_text.append(tgt_entity)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, max_length=args.max_length,
                                              src_lang=args.src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.to('cuda')

    all_log_probs = []
    num_epochs = math.ceil(len(src_text)/args.batch_size)
    for start_idx in tqdm(range(0, num_epochs*args.batch_size, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, len(src_text))
        if args.do_forward:
            input_text = src_text
            output_text = tgt_text
        else:
            input_text = tgt_text
            output_text = src_text
        batch_input_text = input_text[start_idx:end_idx]
        batch_input_text = tokenizer(batch_input_text, return_tensors="pt", max_length=args.max_length,
                                     truncation=True, padding=True)
        batch_input_ids = batch_input_text.input_ids
        batch_attention_mask = batch_input_text.attention_mask

        batch_output_text = output_text[start_idx:end_idx]
        batch_output_text = tokenizer(batch_output_text, max_length=args.max_length, truncation=True,
                                      add_special_tokens=False).input_ids
        batch_src_text_ids = [torch.LongTensor([tokenizer.eos_token_id, tokenizer.lang_code_to_id[args.tgt_lang]] + _ +
                                               [tokenizer.eos_token_id]) for _ in batch_output_text]
        batch_output_ids = pad_sequence(batch_src_text_ids, batch_first=True, padding_value=-100)
        if use_cuda:
            batch_input_ids = batch_input_ids.to('cuda')
            batch_attention_mask = batch_attention_mask.to('cuda')
            batch_output_ids = batch_output_ids.to('cuda')
        lb_scores = enc_dec_scoring(input_ids=batch_input_ids,
                                    target_ids=batch_output_ids,
                                    attention_mask=batch_attention_mask,
                                    model=model)
        log_probs = [_[-1].item() for _ in lb_scores]
        all_log_probs.extend(log_probs)
        # start_idx = end_idx
        # end_idx = min(start_idx + args.batch_size, len(src_text))

    for idx, d in enumerate(normalized_input_data):
        d['reversed_scores'] = []
        if d['flag'] == 0 or d['flag'] == -1:
            continue
        for iidx, partial_text in enumerate(d['src_lang']):
            scores = []
            for iiidx, candidate_text in enumerate(d['tgt_lang'][iidx]):
                src_text.append(partial_text)
                tgt_text.append(candidate_text)
                mapped_idx = idx_mapping[(idx, iidx, iiidx)]
                scores.append(all_log_probs[mapped_idx])
            d['reversed_scores'].append(scores)
    output_path = os.path.join(args.output_path, 'stage4.json')
    logger.info("Save to {}".format(output_path))
    with open(output_path, 'w') as f:
        json.dump(normalized_input_data, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="../pretrained_models/nllb_600m")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--src_lang", type=str, required=True)
    parser.add_argument("--tgt_lang", type=str, default="eng_Latn")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_forward", action="store_true")
    parser.add_argument("--merge_shards", action="store_true")
    args = parser.parse_args()

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_name_or_path
    main(args)
