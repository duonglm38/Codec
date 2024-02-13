CoNLL03_PATH="path/to/CoNLL03/train.txt"  # <- Edit here

OUTPUT_EN_DIR=""  # <- Edit here
OUTPUT_TRANSL_DIR=""  # <- Edit here

if test -f "${OUTPUT_DIR}/conll_en_train_examples.pkl"; then
  echo "conll_en_train_examples.pkl exists, proceed to the next step"
else
  python pipelines/process_en_data.py \
                    --input_path ${CoNLL03_PATH} \
                    --output_dir ${OUTPUT_DIR}
fi

ORG_SENT_PATH="${OUTPUT_DIR}/conll_en_train_org.txt"
ENTITY_PATH="${OUTPUT_DIR}/conll_en_train_entity.txt"
MODEL_NAME_OR_PATH="facebook/nllb-200-3.3B"

for idx in {1..18}; do
  TGT_LANG=$(sed -n "${idx}p" masakha_lang_ids.txt)
  python pipelines/nllb_translation_tasks.py \
                              --source_language eng_Latn \
                              --target_language $TGT_LANG \
                              --model_name_or_path $MODEL_NAME_OR_PATH \
                              --tokenizer_path $MODEL_NAME_OR_PATH \
                              --input_file_path $ORG_SENT_PATH \
                              --output_folder $OUTPUT_TRANSL_DIR \
                              --output_fname "conll_en_train_org_${TGT_LANG}.txt" \
                              --shard_num -1

  sacremoses -l en -j 4 tokenize  < "$OUTPUT_TRANSL_DIR/conll_en_train_org_${TGT_LANG}.txt" > "$OUTPUT_TRANSL_DIR/conll_en_train_org_${TGT_LANG}.txt.tok"

  python pipelines/nllb_translation_tasks.py \
                              --source_language eng_Latn \
                              --target_language $TGT_LANG \
                              --model_name_or_path $MODEL_NAME_OR_PATH \
                              --tokenizer_path $MODEL_NAME_OR_PATH \
                              --input_file_path $ENTITY_PATH \
                              --output_folder $OUTPUT_TRANSL_DIR \
                              --output_fname "conll_en_train_entity_${TGT_LANG}.txt" \
                              --shard_num -1
done
