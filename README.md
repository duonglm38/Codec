# Constrained Decoding for Cross-lingual Label Projection

This repo contains the code for our ICLR 2024 paper: <a href="https://arxiv.org/abs/2402.03131"> Constrained Decoding for Cross-lingual Label Projection.</a>

## Installation
This project uses python 3.9.16
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
```

## Quick usage
> [!important]
> We have noticed strange behaviors in the `transformers` tokenizer when loading from finetuned NLLB checkpoints. To reproduce the results in the paper, please load the tokenizer from the original NLLB checkpoints (e.g., `facebook/nllb-200-distilled-600M`).

We have provided a simple code to run Codec in `example.py`: 
 ```
 MODEL="ychenNLP/nllb-200-distilled-1.3B-easyproject"
 NLLB_TOKENIZER="facebook/nllb-200-distilled-1.3B"  # <--- Load from the original NLLB checkpoint

 SRC_TEXT="Only France and [ Britain ] backed Fischler 's proposal ."
 TEMPLATE="Faransi ni Angiletɛri dɔrɔn de ye Fischler ka laɲini dɛmɛ ."
 
 python example.py \
            --src_text ${SRC_TEXT} \
            --template ${TEMPLATE} \
            --model_name_or_path ${MODEL} \
            --tokenizer_path ${NLLB_TOKENIZER}
 ```
The current codebase supports mBART, M2M-100, and NLLB model checkpoints. The fine-tuned version of NLLB-600M can be downloaded <a href="https://drive.google.com/file/d/1huu8QuzlbGwkbXfn9_xiWfUNY_B7ePaj/view?usp=sharing"> here </a> (please load the NLLB tokenizer from `facebook/nllb-200-distilled-600M` when using this checkpoint).


## Cross-lingual NER

### Translate train

* Download the CoNLL-2003 and MasakhaNER2.0 datasets
* Run the `scripts/translate_conll.sh` script, you will need to edit the input and output path inside the script first.
  * This script will process the CoNLL03 dataset and translate the training data to 18 African languages
* Run the script `scripts/augment_ner.sh`, you will need to edit the target language, input and output paths first
  * This script will prepare the input to Codec, run Codec, and create the augmented training data on the target language
* To train the NER model, we use the code and script from <a href="https://github.com/edchengg/easyproject/tree/main/ner#ner-training"> this repo </a>

### Translate test

* Train an English NER model on CoNLL-2003 dataset (code: <a href="https://github.com/edchengg/easyproject/tree/main/ner#ner-training"> this repo </a>). In CoNLL-2003, we convert all `MISC` to `O` tag before training.
* Translate the test data to English (code: `pipelines/nllb_translation_tasks.py`), than use the English NER model to annotate.
* Run the script `scripts/masakhaner_translate_test.sh`, you will need to edit the target language, input and output paths first

## Citation
If you use this codebase in your work, please consider citing our paper:
```
@inproceedings{
le2024constrained,
title={Constrained Decoding for Cross-lingual Label Projection},
author={Duong Minh Le and Yang Chen and Alan Ritter and Wei Xu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=DayPQKXaQk}
}
```
