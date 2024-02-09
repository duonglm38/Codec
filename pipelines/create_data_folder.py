import os
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--conll_2003_path", type=str, default="conll_data/en/conll_2003/")
parser.add_argument("--parsed_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--language", type=str, required=True)
parser.add_argument("--suffix", type=str, default='')
parser.add_argument("--add_en", action="store_true")
args = parser.parse_args()

with open("{}/train.txt".format(args.conll_2003_path), "r", encoding="utf-8") as f:
    en_lines = f.readlines()

if args.add_en:
    type_mark = "marker"
else:
    type_mark = "marker-proj"
for lang in [args.language]:

    lang = lang.split("_")[0]
    output_folder = "{}/en-{}-{}{}".format(args.output_dir, lang, type_mark, args.suffix)
    if os.path.exists("{}/train.txt".format(output_folder)):
        print("Data folder exist!")
        break
    with open(args.parsed_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # lines = open(args.parsed_file, "r", encoding="utf-8").readlines()

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open("{}/train.txt".format(output_folder), "w", encoding="utf-8") as f:
        if args.add_en:
            for l in en_lines:
                f.write(l)
            f.write("\n")
        if lang not in ["deu", "spa", "nld"]:
            for l in lines:
                # remove MISC
                l = l.replace("B-MISC", "O")
                l = l.replace("I-MISC", "O")
                f.write(l)

    # f = open("{}/en-{}-{}/train.txt".format(output_folder, lang, type_mark), "w", encoding="utf-8")
    # for l in en_lines:
    #     f.write(l)
    # f.write("\n")

    src_path = "{}/dev.txt".format(args.conll_2003_path)
    dst_path = "{}/dev.txt".format(output_folder)
    shutil.copy(src_path, dst_path)

    src_path = "{}/test.txt".format(args.conll_2003_path)
    dst_path = "{}/test.txt".format(output_folder)
    shutil.copy(src_path, dst_path)
