import code
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
from utils import tokenize_non_whitespace

THRESHOLD = 106
THRESHOLD2 = 170
THRESHOLD3 = 200


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
        input_data = json.load(f)

    if args.shard_num is not None:
        input_data = list(split(input_data, args.total_shard))[args.shard_num]

    print("Decode (iterative 1) {} examples:....".format(len(input_data)))
    acc_time = 0
    results = []
    decoded_token_ids = []
    for i in tqdm(range(len(input_data))):
        template_text = input_data[i]['template']
        if args.tgt_lang.startswith('zh'):
            pre_template_ids = tokenize_non_whitespace(template_text, tokenizer)
        else:
            pre_template_ids = tokenizer(template_text, max_length=args.max_length, truncation=True,
                                         add_special_tokens=False).input_ids
        template_ids = [tokenizer.eos_token_id, tokenizer.lang_code_to_id[args.tgt_lang]] + pre_template_ids + \
                       [tokenizer.eos_token_id]
        template_ids = torch.LongTensor(template_ids)
        white_space_positions = set()
        if args.white_space_only:
            template_tokens = tokenizer.tokenize(template_text)
            for token_idx, token in enumerate(template_tokens):
                if token.startswith('▁'):
                    white_space_positions.add(token_idx + 2)
        input_text_list, output_text_list, output_text_ids_list, output_acc_log_prob_list, score_list = [], [], [], [], []
        all_tgt_entities, all_marker_positions = [], []
        flag = input_data[i]['flag']

        if flag == 0 or flag == -1:
            item = {
                "idx": i,
                "src_lang": input_text_list,
                "template": template_text,
                "tgt_lang": output_text_list,
                "score": score_list,
                "flag": flag
            }
            results.append(item)
            decoded_token_ids.append(None)
            continue
        for ii, text_to_decode in enumerate(input_data[i]['text_to_decode']):
            partial_marker_text = text_to_decode['text']
            possible_opening_position = text_to_decode['candidates']
            possible_closing_position = None
            if 'right_candidates' in text_to_decode:
                right_candidates = text_to_decode['right_candidates']
                possible_closing_position = dict()
                for idx, k in enumerate(possible_opening_position):
                    possible_closing_position[k] = set(right_candidates[idx])
            if len(possible_opening_position) == 0:
                possible_opening_position = None
            else:
                possible_opening_position = set(possible_opening_position)
            if args.not_use_candidate:
                possible_opening_position = None
                possible_closing_position = None
            if args.not_use_right_candidate:
                possible_closing_position = None
            if white_space_positions:
                if possible_opening_position:
                    possible_opening_position = possible_opening_position.intersection(white_space_positions)
                else:
                    possible_opening_position = white_space_positions
                possible_closing_position = dict()
                for k in possible_opening_position:
                    possible_closing_position[k] = white_space_positions

            input_ids = tokenizer(partial_marker_text, return_tensors="pt", max_length=args.max_length,
                                  truncation=True).input_ids
            if use_cuda:
                input_ids = input_ids.to('cuda')
            if not args.use_round_marker:
                bracket_stack = [']', '[']
                left_marker = '['
                right_marker = ']'
            else:
                bracket_stack = [')', '(']
                left_marker = '('
                right_marker = ')'
            effective_search_mode = args.search_mode
            effective_batch_size = args.batch_size
            if len(template_ids) > THRESHOLD:
                effective_search_mode = 1
                effective_batch_size = args.batch_size*3//4
            if len(template_ids) >= THRESHOLD2:
                effective_batch_size = args.batch_size//2
            if len(template_ids) > THRESHOLD3:
                effective_batch_size = args.batch_size//4
            decode_args = BracketConstraintDecodingArgument(
                template_ids=template_ids,
                bracket_stack=bracket_stack,
                template_pointer=1,
                model_name=args.mt_name,
                future_steps=args.future_steps,
                search_mode=effective_search_mode,
                batch_size=effective_batch_size,
                save_visualization=args.save_visualization,
                n_best=args.n_best,
                possible_opening_positions=possible_opening_position,
                possible_closing_positions=possible_closing_position,
                left_marker=left_marker,
                right_marker=right_marker
            )

            start = timer()
            outputs = generate(self=model,
                               inputs=input_ids,
                               decoding_argument=decode_args.arguments,
                               forced_bos_token_id=tokenizer.lang_code_to_id[args.tgt_lang],
                               max_length=args.max_length,
                               num_beams=args.num_beams, length_penalty=0)
            end = timer()
            acc_time += end - start
            input_text_list.append(partial_marker_text)
            if len(outputs.min_heap) == 0:
                output_text_ids_list.append(template_ids.unsqueeze(0).cpu())
                all_marker_positions.append([[0, 0]])
                output_text_list.append([template_text])
                all_tgt_entities.append([''])
                score_list.append([-float('inf')])
            else:
                outputs.min_heap.sort(reverse=True)
                all_candidate_scores, all_candidate_text_ids, all_candidate_acc_log_probs = [], [], []
                candidate_entities = []
                marker_positions = []
                for candidate in outputs.min_heap:
                    left_marker_position, right_marker_position = candidate[4]
                    # assert candidate[2][0][left_marker_position] in [104, 709] and \
                    #        candidate[2][0][right_marker_position] in [10109, 14229]
                    marker_positions.append([left_marker_position, right_marker_position])
                    if left_marker_position < right_marker_position - 1:
                        entity_ids = candidate[2][0][left_marker_position + 1:right_marker_position]
                        candidate_entities.append(tokenizer.decode(entity_ids))
                    else:
                        candidate_entities.append('')

                    all_candidate_scores.append(candidate[0])
                    all_candidate_text_ids.append(candidate[2])
                    all_candidate_acc_log_probs.append(candidate[3])

                all_candidate_text_ids = torch.cat(all_candidate_text_ids, dim=0)
                output_text_list.append(tokenizer.batch_decode(all_candidate_text_ids, skip_special_tokens=True))
                output_text_ids_list.append(all_candidate_text_ids.cpu())
                output_acc_log_prob_list.append(all_candidate_acc_log_probs)
                score_list.append(all_candidate_scores)
                all_tgt_entities.append(candidate_entities)
                all_marker_positions.append(marker_positions)
        decoded_token_ids.append([output_text_ids_list, all_marker_positions])
        item = {
            "idx": i,
            "src_lang": input_text_list,
            "src_entities": input_data[i]['src_entities'],
            "template": template_text,
            "tgt_lang": output_text_list,
            "tgt_entities": all_tgt_entities,
            "score": score_list,
            "flag": flag
        }

        results.append(item)

    print("Total running time:", acc_time)
    print("Avg time per sample:", acc_time/len(input_data))
    if args.time_log:
        content = f"Total running time: {acc_time}\nAvg time per sample: {acc_time/len(input_data)}\n"
        with open(args.time_log, 'w') as f:
            f.write(content)
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
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--src_lang", type=str, default="eng_Latn")
    parser.add_argument("--tgt_lang", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--shard_num", type=int, default=None)
    parser.add_argument("--total_shard", type=int, default=10)
    parser.add_argument("--search_mode", choices=[0, 1, 2], type=int, default=0, help="0: fast heuristic,"
                                                                                      "1: slow heuristic"
                                                                                      "2: Constrained BS"
                        )
    parser.add_argument("--n_best", type=int, default=5)
    parser.add_argument("--num_beams", type=int, default=5, help="Beam size for Constrained BS")
    parser.add_argument("--future_steps", default=-1, type=int)
    parser.add_argument("--save_visualization", action="store_true")
    parser.add_argument("--search_tree_path", default=None, type=str)
    parser.add_argument("--not_use_candidate", type=int, choices=[0, 1], default=0)
    parser.add_argument("--not_use_right_candidate", action="store_true")
    parser.add_argument("--white_space_only", action="store_true")
    parser.add_argument("--use_round_marker", action="store_true")
    parser.add_argument("--mt_name", type=str, default='nllb')
    parser.add_argument("--time_log", type=str, default=None)
    args = parser.parse_args()

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_name_or_path
    not_use_candidate = args.not_use_candidate
    if not_use_candidate == 0:
        args.not_use_candidate = False
    else:
        args.not_use_candidate = True
    print("not_use_candidate:", args.not_use_candidate)
    main(args)
