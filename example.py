import argparse
import torch
from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
    )
from decoding_argument import BracketConstraintDecodingArgument
from generation_utils import generate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_text", type=str, default="Only France and [ Britain ] backed Fischler 's proposal .")
    parser.add_argument("--template", type=str, default="Faransi ni Angiletɛri dɔrɔn de ye Fischler ka laɲini dɛmɛ .")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    args = parser.parse_args()
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_name_or_path

    src_text = args.src_text
    template_text = args.template
    tgt_lang = "bam_Latn"

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenized_input = tokenizer(src_text, return_tensors="pt")
    model.to('cuda')
    input_ids = tokenized_input.input_ids
    input_ids = input_ids.to('cuda')
    tmp = []

    template_ids = tokenizer.encode(template_text, add_special_tokens=False)
    template_ids = [tokenizer.eos_token_id, tokenizer.lang_code_to_id[tgt_lang]] + template_ids + \
                   [tokenizer.eos_token_id]

    template_ids = torch.LongTensor(template_ids)
    bracket_stack = [']', '[']
    template_length = template_ids.shape[-1] - 3
    print("Input:", src_text)
    print("template:", template_text)
    print("Template length:", template_length)

    possible_opening_positions = None  # set(possible_opening_positions)
    decode_args = BracketConstraintDecodingArgument(
        template_ids=template_ids,
        bracket_stack=bracket_stack,
        template_pointer=1,
        model_name='nllb',
        future_steps=5,
        search_mode=0,
        batch_size=16,
        n_best=5,
        possible_opening_positions=possible_opening_positions,
        left_marker='[',
        right_marker=']'
    )

    outputs = generate(self=model,
                       inputs=input_ids,
                       decoding_argument=decode_args.arguments,
                       forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                       num_beams=4, length_penalty=0
                       )
    outputs.min_heap.sort(reverse=True)
    for candidate in outputs.min_heap:
        print(tokenizer.batch_decode(candidate[2], skip_special_tokens=True)[0])  # , skip_special_tokens=True))
        print("Score:", candidate[0])
        print("-------------")
