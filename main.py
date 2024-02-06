import os
import pickle
from timeit import default_timer as timer
import torch
import argparse
import json
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from decoding_argument import BracketConstraintDecodingArgument
from generation_utils import generate
from sacremoses import MosesTokenizer, MosesDetokenizer
from utils import preprocess, compute_number_combination


def split(a, n):
    """split a list in to n equally sized chunks."""
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, max_length=args.max_length,
                                              src_lang=args.src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.to('cuda')

    with open(args.input_path) as f:
        input_text = f.read().strip().splitlines()
    with open(args.original_text_path) as f:
        original_text = f.read().strip().splitlines()
    with open(args.template_path) as f:
        template_text = f.read().strip().splitlines()

    if args.shard_num is not None:
        input_text = list(split(input_text, 10))[args.shard_num]
        original_text = list(split(original_text, 10))[args.shard_num]
        template_text = list(split(template_text, 10))[args.shard_num]
    print("Decode {} examples:....".format(len(input_text)))
    if args.is_tokenized:
        src_mt, src_md = MosesTokenizer(lang=args.moses_src_lang), MosesDetokenizer(lang=args.moses_src_lang)
        tgt_mt, tgt_md = MosesTokenizer(lang=args.moses_tgt_lang), MosesDetokenizer(lang=args.moses_tgt_lang)

        input_text = [preprocess(src_mt, src_md, _.split(" "), is_tokenized=True) for _ in input_text]
        template_text = [preprocess(tgt_mt, tgt_md, _.split(" "), is_tokenized=True) for _ in template_text]

    # Find number of additional brackets
    num_additional_brackets = []
    for bracket_text, org_text in zip(input_text, original_text):
        num_open_bracket = bracket_text.count('[') - org_text.count('[')
        num_close_bracket = bracket_text.count(']') - org_text.count(']')
        num_additional_brackets.append(min(num_close_bracket, num_open_bracket))

    acc_time = 0
    results = []
    decoded_token_ids = []
    for i in tqdm(range(len(input_text))):
        if -1 < args.test_mode == i:
            break
        input_ids = tokenizer(input_text[i], return_tensors="pt", max_length=args.max_length, truncation=True).input_ids
        pre_template_ids = tokenizer(template_text[i], max_length=args.max_length, truncation=True,
                                     add_special_tokens=False).input_ids
        template_ids = [tokenizer.eos_token_id, tokenizer.lang_code_to_id[args.tgt_lang]] + pre_template_ids + \
                       [tokenizer.eos_token_id]
        template_ids = torch.LongTensor(template_ids)
        if use_cuda:
            input_ids = input_ids.to('cuda')
        num_additional_bracket = num_additional_brackets[i]
        bracket_stack = [']', '[']*num_additional_bracket
        decode_args = BracketConstraintDecodingArgument(
            template_ids=template_ids,
            bracket_stack=bracket_stack,
            template_pointer=1,
            model_name='nllb',
            future_steps=args.future_steps,
            search_mode=args.search_mode,
            batch_size=args.batch_size,
            save_visualization=args.save_visualization
        )

        start = timer()
        outputs = generate(self=model,
                           inputs=input_ids,
                           decoding_argument=decode_args.arguments,
                           forced_bos_token_id=tokenizer.lang_code_to_id[args.tgt_lang],
                           max_length=args.max_length)
        end = timer()
        acc_time += end - start
        output_text = tokenizer.batch_decode(outputs.text_ids, skip_special_tokens=True)[0]
        decoded_token_ids.append([outputs.text_ids.cpu(), outputs.accumulate_scores])
        score = outputs.score.item()
        item = {
            "src_lang": input_text[i],
            "template": template_text[i],
            "tgt_lang": output_text,
            "score": score
        }
        results.append(item)
        if args.save_visualization and outputs.search_tree is not None and args.search_tree_path:
            out_tree_path = os.path.join(args.search_tree_path, 'trees')
            if not os.path.exists(out_tree_path):
                os.mkdir(out_tree_path)
            out_tree_file = os.path.join(out_tree_path, 'tree_{}.pkl'.format(i))
            with open(out_tree_file, 'wb') as f:
                pickle.dump(outputs.search_tree, f)

    print("Total running time:", acc_time)
    print("Avg time per sample:", acc_time/len(input_text))
    if args.output_path is not None:
        output_path = args.output_path
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        torch.save(decoded_token_ids, output_path.replace('.json', '.pt'))
    else:
        print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="../pretrained_models/nllb_600m")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--original_text_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--template_path", type=str, required=True)
    parser.add_argument("--src_lang", type=str, default="eng_Latn")
    parser.add_argument("--tgt_lang", type=str, required=True)
    parser.add_argument("--moses_src_lang", type=str, default="en")
    parser.add_argument("--moses_tgt_lang", type=str, default=None)
    parser.add_argument("--is_tokenized", action="store_true")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--shard_num", type=int, default=None)
    parser.add_argument("--search_mode", choices=[0, 1], type=int, default=0, help="0: heuristic, 1: exact")
    parser.add_argument("--future_steps", default=-1, type=int)
    parser.add_argument("--test_mode", type=int, default=-1, help="generate the first n sentences")
    parser.add_argument("--visit_threshold", default=1., type=float)
    parser.add_argument("--save_visualization", action="store_true")
    parser.add_argument("--search_tree_path", default=None, type=str)
    args = parser.parse_args()

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_name_or_path
    main(args)
