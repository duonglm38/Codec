DATADIR="conll_data_nllb3b"
MODEL="nllb600mft"
MODEL_PATH=""
TOKENIZER_PATH=""
TGTLANG="bam"
FULL_TGTLANG="bam_Latn"  # <- language code in NLLB
LANG_MOSES="${TGTLANG}"
FUTURE_STEP=1   # lower bound hyperparameter

CoNLL03_PATH="path/to/CoNLL03"

# The 5 below files are extracted by the `translate_conll.sh` script
ORIGINAL_FILE=""   # <- file contains sentences from CoNLL03, one sentence per line
ENTITY_TRANS_PATH=""   # <- file contain translation of each entity from CoNLL03, one entity per line
TEMPLATE_FILE=""   # <- file contains translation of the ORIGINAL_FILE, one sentence per line
TOK_TEMPLATE_FILE=""   # <- file contains tokenization of the TEMPLATE_FILE
CONLL_PKL_FILE=""  # <- path to the `conll_en_train_examples.pkl` file

OUTPUT_DIR="outputs/${LANG_MOSES}"
OUTPUT_DIR1="${OUTPUT_DIR}/${MODEL}_preprocess"
OUTPUT_DIR2="${OUTPUT_DIR}/${MODEL}_iter_${FUTURE_STEP}/search_results"
OUTPUT_DIR3="${OUTPUT_DIR}/${MODEL}_iter_${FUTURE_STEP}"
OUTPUT_DIR4="path/to/output/augmented/data"


### Preprocess
if test -d $OUTPUT_DIR1; then
  echo ""
else
  mkdir -p $OUTPUT_DIR1
fi

echo "Preprocessing: ${TGTLANG} / ${FULL_TGTLANG}"

INPUT_FILE_PATH="${CoNLL03_PATH}/train.txt"
python pipelines/stage1_from_conll.py \
                  --model_name_or_path ${MODEL_PATH} \
                  --tokenizer_path ${TOKENIZER_PATH} \
                  --input_path ${INPUT_FILE_PATH} \
                  --template_path ${TOK_TEMPLATE_FILE} \
                  --org_template_path ${TEMPLATE_FILE} \
                  --tgt_lang ${FULL_TGTLANG} \
                  --batch_size 32 \
                  --output_path ${OUTPUT_DIR1} \
                  --is_tokenized

python pipelines/stage2.py \
                 --stage1_path ${OUTPUT_DIR1} \
                 --output_path ${OUTPUT_DIR1} \
                 --threshold1 0.5

### Decoding
INPUT_FILE_PATH="${OUTPUT_DIR1}/stage2.json"
if test -d $OUTPUT_DIR2; then
  echo ""
else
  mkdir -p $OUTPUT_DIR2
fi

for shard in {0..9}; do
    echo "Decoding shard #${shard}"
    output_file_path="${OUTPUT_DIR2}/conll_en_train_decode_marker.${FULL_TGTLANG}.${shard}.json"
    python iterative_decode.py \
                 --model_name_or_path "${MODEL_PATH}" \
                 --tokenizer_path "${TOKENIZER_PATH}" \
                 --input_path "${INPUT_FILE_PATH}" \
                 --output_path ${output_file_path} \
                 --tgt_lang ${FULL_TGTLANG} \
                 --batch_size 16 \
                 --shard_num ${shard} \
                 --max_length 156 \
                 --future_steps ${FUTURE_STEP} \
                 --n_best 5


### Postprocessing
python pipelines/stage4.py \
                 --model_name_or_path ${MODEL_PATH} \
                 --tokenizer_path ${TOKENIZER_PATH} \
                 --input_path ${OUTPUT_DIR2} \
                 --output_path ${OUTPUT_DIR3} \
                 --src_lang ${FULL_TGTLANG} \
                 --merge_shards \
                 --batch_size 256

python pipelines/stage5.py \
                --input_file ${OUTPUT_DIR3}/stage4.json \
                --output_path ${OUTPUT_DIR3} \
                --entity_trans_path ${ENTITY_TRANS_PATH} \
                --conll_en_pkl_path ${CONLL_PKL_FILE} \
                --tgt_lang ${LANG_MOSES} \
                --mode 4 \
                --tokenizer_path ${TOKENIZER_PATH} \
                --search_log_path ${OUTPUT_DIR2}


### Create train/dev/test file
python pipelines/create_data_folder.py \
                  --conll_2003_path ${CoNLL03_PATH} \
                  --parsed_file ${OUTPUT_DIR3}/stage5_mode-4.txt \
                  --output_dir ${OUTPUT_DIR4} \
                  --language ${LANG_MOSES} \
                  --suffix "_future${FUTURE_STEP}_mode-4" \
                  --add_en

