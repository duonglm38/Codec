"""
Translation script for EasyProject - all task data is in txt format
Input: a txt file
Output: a txt file
transformers              4.25.1 
torch                     1.11.0+cu113             pypi_0    pypi
"""
import argparse
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBartForConditionalGeneration, MBart50TokenizerFast
import torch
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--target_language", default=None, type=str, required=True,
                help="target language code.")
    parser.add_argument("--source_language", default="eng_Latn", type=str, required=False,
                help="source language code.")
    parser.add_argument("--batch_size", default=8, type=int, required=False,
                help="batch size.")
    parser.add_argument("--max_length", default=128, type=int, required=False,
                help="max_length.")
    parser.add_argument("--shard_num", default=0, type=int, required=False,
                help="shard idx.")
    parser.add_argument("--total_shard", default=2, type=int, required=False,
                            help="total number of shards.")
    parser.add_argument("--output_folder", default="", type=str, required=True,
                help="input.")
    parser.add_argument("--output_fname", default=None, type=str, help="output file name")
    parser.add_argument("--input_file_path", default="", type=str, required=True,
                help="input.")
    parser.add_argument("--model_name_or_path", default="facebook/nllb-200-3.3B", type=str, required=True, 
                help="model name or path.")
    parser.add_argument("--tokenizer_path", default="facebook/nllb-200-3.3B", type=str, required=True,
                help="tokenizer path.")
    parser.add_argument("--num_beams", default=5, type=int, required=False,
                            help="beam size.")

    args = parser.parse_args()
    batch_size = args.batch_size
    
    # load data
    input_lines = []
    with open(args.input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            input_lines.append(line.strip())

    print(len(input_lines))

    def split(a, n):
        """split a list in to n equally sized chunks."""
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    
    # shard into 2 splits  and select the shard_num piece 
    input_chunks = input_lines
    if args.shard_num >= 0: 
        input_chunks = list(split(input_lines, args.total_shard))[args.shard_num]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, src_lang=args.source_language)
    # tokenizer = MBart50TokenizerFast.from_pretrained(args.tokenizer_path, src_lang=args.source_language)
    print("Loading model")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    # model = MBartForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.cuda()
    
    print("Start translation...")
    output_result = []
    for idx in tqdm(range(0, len(input_chunks), batch_size)):
        start_idx = idx
        end_idx = idx + batch_size
        inputs = tokenizer(input_chunks[start_idx: end_idx], padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to('cuda')

        with torch.no_grad():
            translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[args.target_language], max_length=args.max_length, num_beams=5, num_return_sequences=1, early_stopping=True)
        output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        output_result.extend(output)
    
    # save
    file_name = args.input_file_path.split("/")[-1]
    if args.output_fname:
        save_path = os.path.join(args.output_folder, args.output_fname)
    else:
        save_path = os.path.join(args.output_folder, "{}.{}.{}".format(file_name, args.target_language, args.shard_num))
    with open(save_path, "w", encoding="utf-8") as f:
        for out in output_result:
            f.write(out + "\n")
