## Cross-lingual NER

### Translate train

* Download the CoNLL-2003 and MasakhaNER2.0 datasets
* Run the `translate_conll.sh` script, you will need to edit the input and output path inside the script first.
  * This script will process the CoNLL03 dataset and translate the training data to 18 African languages
* Run the script `augment_ner.sh`, you will need to edit the target language, input and output paths first
  * This script will prepare the input to Codec, run Codec, and create the augmented training data on the target language
* To train the NER model, we use the code and script from <a href="https://github.com/edchengg/easyproject/tree/main/ner#ner-training"> this repo </a>