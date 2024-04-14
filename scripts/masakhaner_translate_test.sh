MODEL="nllb600mft"
MODEL_PATH=""
TOKENIZER_PATH=""
DATADIR=""
TGTLANG="bam"
FULL_TGTLANG="bam_Latn"  # <- language code in NLLB
LANG_MOSES="${TGTLANG}"
SEED=1
STAGE5_MODE=1
FUTURE_STEP=5   # lower bound hyperparameter

OUTPUT_DIR="translate_test_outputs/${LANG_MOSES}"
OUTPUT_DIR1="${OUTPUT_DIR}/${MODEL}_preprocess"
OUTPUT_DIR2="${OUTPUT_DIR}/${MODEL}_iter_${FUTURE_STEP}/search_results"
OUTPUT_DIR3="${OUTPUT_DIR}/${MODEL}_iter_${FUTURE_STEP}"

### Preprocess
if test -d $OUTPUT_DIR1; then
  echo ""
else
  mkdir -p $OUTPUT_DIR1
fi

echo "Preprocessing: ${TGTLANG} / ${FULL_TGTLANG}"

MASAKHANER_PATH=""  # <- path to the data folder of the language in MasakhaNER2.0 (e.g., `bam`)
INPUT_FILE_PATH=""  # <- path to the output of the English NER model after annotating the translation of test file (example: `examples/bam_test_prediction_sample.txt`)
TOK_TEMPLATE_FILE=""  # <- path to text of the test file, one example per line (example: `examples/bam_test_sample.txt`)
TEMPLATE_FILE="${TOK_TEMPLATE_FILE}"

if test -f "${OUTPUT_DIR1}/stage1.json"; then
    echo "Stage 1: File existed"
    echo ""
else
    python pipelines/stage1_from_conll.py \
                         --model_name_or_path ${MODEL_PATH} \
                         --tokenizer_path ${TOKENIZER_PATH} \
                         --input_path ${INPUT_FILE_PATH} \
                         --template_path ${TOK_TEMPLATE_FILE} \
                         --org_template_path ${TEMPLATE_FILE} \
                         --tgt_lang ${FULL_TGTLANG} \
                         --batch_size 32 \
                         --output_path ${OUTPUT_DIR1} \
                         --no_filtering
fi

if test -f "${OUTPUT_DIR1}/stage2.json"; then
    echo "Stage 2: File existed"
    echo ""
else
    python ${WORKDIR}/processing/stage2.py \
                         --stage1_path ${OUTPUT_DIR} \
                         --output_path ${OUTPUT_DIR} \
                         --threshold1 0.5
fi

### Decoding
INPUT_FILE_PATH="${OUTPUT_DIR1}/stage2.json"
if test -d $OUTPUT_DIR2; then
  echo ""
else
  mkdir -p $OUTPUT_DIR2
fi

for shard in {0..3}; do
    echo "Decoding shard #${shard}"
    output_file_path="${OUTPUT_DIR2}/conll_en_train_decode_marker.${FULL_TGTLANG}.${shard}.json"
    if test -f "${output_file_path}"; then
        echo "File ${output_file_path} exists"
    else
        python ${WORKDIR}/iterative_decode.py \
                       --model_name_or_path "${MODEL_PATH}" \
                       --tokenizer_path "${TOKENIZER_PATH}" \
                       --input_path "${INPUT_FILE_PATH}" \
                       --output_path ${output_file_path} \
                       --tgt_lang ${FULL_TGTLANG} \
                       --batch_size 16 \
                       --shard_num ${shard} \
                       --max_length 156 \
                       --future_steps ${FUTURE_STEP} \
                       --n_best 5 \
                       --total_shard 4
    fi
done

### Postprocessing
if test -f "${OUTPUT_DIR3}/stage4.json"; then
    echo "Stage 4: File existed"
    echo ""
else
    python ${WORKDIR}/processing/stage4.py \
                           --model_name_or_path ${MODEL_PATH} \
                           --tokenizer_path ${TOKENIZER_PATH} \
                           --input_path ${OUTPUT_DIR2} \
                           --output_path ${OUTPUT_DIR3} \
                           --src_lang ${FULL_TGTLANG} \
                           --merge_shards \
                           --batch_size 256
fi

if test -f "${OUTPUT_DIR}/stage5_mode-${STAGE5_MODE}${SUFFIX}.txt"; then
    echo "Stage 5: File existed"
    echo ""
else
    python ${WORKDIR}/processing/stage5_ts.py \
                          --input_file ${OUTPUT_DIR3}/stage4.json \
                          --output_path ${OUTPUT_DIR3} \
                          --en_label_path ${INPUT_FILE_PATH} \
                          --masakhaner_path ${MASAKHANER_PATH} \
                          --mode ${STAGE5_MODE} \
                          --tokenizer_path ${TOKENIZER_PATH}
fi
