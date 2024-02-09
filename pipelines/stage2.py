import os
import sys
sys.path.append('./')

import argparse
import logging
from tqdm import tqdm
import torch
import json


log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger = logging.getLogger()


def get_transition_log_probs(accumulate_score):
    pad_tensor = torch.zeros_like(accumulate_score)
    pad_tensor[1:] = accumulate_score[:-1]
    return accumulate_score - pad_tensor


def get_candidates(pre_transition_probs, post_transition_probs, threshold1, threshold2, window_size=5):
    delta = (post_transition_probs - pre_transition_probs).abs()
    candidate_positions = (delta >= threshold1).nonzero().squeeze(1)
    final_candidate_list = set()
    for position in candidate_positions:
        start_idx = max(0, position - window_size)
        end_idx = min(len(pre_transition_probs), position + window_size + 1)
        candidates = [_ for _ in range(start_idx, end_idx) if delta[_] >= threshold2]
        final_candidate_list.update(candidates)
    score_position_pairs = [[delta[_].item(), _] for _ in final_candidate_list]
    # score_position_pairs = sorted(score_position_pairs, reverse=True)
    final_candidate_list = [_[1] + 2 for _ in score_position_pairs]
    return final_candidate_list


def main(args):
    logger.info("Stage 2:......................")
    input_text_path = os.path.join(args.stage1_path, "stage1.json")
    log_prob_path = os.path.join(args.stage1_path, "stage1.pt")
    with open(input_text_path) as f:
        input_text_data = json.load(f)
    acc_log_probs = torch.load(log_prob_path)
    j = 0

    data = []
    for idx, d in enumerate(tqdm(input_text_data)):
        src_log_prob = acc_log_probs[j]
        src_transition_log_probs = get_transition_log_probs(src_log_prob)
        j += 1
        item = []
        for _ in range(len(d['text_to_decode'])):
            tgt_transition_log_probs = get_transition_log_probs(acc_log_probs[j])
            position_candidates = get_candidates(pre_transition_probs=src_transition_log_probs,
                                                 post_transition_probs=tgt_transition_log_probs,
                                                 threshold1=args.threshold1,
                                                 threshold2=args.threshold2,
                                                 window_size=args.window_size)
            item.append({
                'text': d['text_to_decode'][_],
                'candidates': position_candidates
            })
            j += 1
        data.append({
            'src_lang': d['src_org_text'],
            'src_entities': d['src_entities'],
            'template': d['template'],
            'text_to_decode': item,
            "flag": d['flag']
        })
    output_path = os.path.join(args.output_path, "stage2{}.json".format(args.suffix))
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Save data to: {}".format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--threshold1", type=float, default=0.5)
    parser.add_argument("--threshold2", type=float, default=0.1)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    main(args)
