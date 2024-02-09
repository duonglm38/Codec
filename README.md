# Constrained Decoding for Cross-lingual Label Projection

This repo contains the code for our ICLR 2024 paper: <a href="https://arxiv.org/abs/2402.03131"> Constrained Decoding for Cross-lingual Label Projection.</a>

## Installation
This project uses python 3.9.16
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
```

## Quick usage
We have provided a simple code to run Codec in `example.py`: 
 ```
 MODEL="ychenNLP/nllb-200-distilled-1.3B-easyproject"
 SRC_TEXT="Only France and [ Britain ] backed Fischler 's proposal ."
 TEMPLATE="Faransi ni Angiletɛri dɔrɔn de ye Fischler ka laɲini dɛmɛ ."
 
 python example.py \
            --src_text ${SRC_TEXT} \
            --template ${TEMPLATE} \
            --model_name_or_path ${MODEL}  
 ```
The current codebase supports mBART, M2M-100, and NLLB model checkpoints


## Citation
If you use this codebase in your work, please consider citing our paper:
```
@article{le2024constrained,
  title={Constrained Decoding for Cross-lingual Label Projection},
  author={Le, Duong Minh and Chen, Yang and Ritter, Alan and Xu, Wei},
  journal={arXiv preprint arXiv:2402.03131},
  year={2024}
}
```