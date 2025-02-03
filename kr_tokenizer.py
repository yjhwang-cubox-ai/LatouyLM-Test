from transformers import AutoProcessor, AutoTokenizer
import json
from PIL import Image
import os

korean_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

processor = AutoProcessor.from_pretrained(
    "microsoft/layoutlmv2-base-uncased",
    tokenizer = korean_tokenizer,
    revision="no_ocr"
)

with open("KR_ID_LAYOUT_TRAIN/annotations/id_00001.json", "r", encoding="utf-8") as f:
    data = json.load(f)

word_data = data['entities']
words = [text['text'] for text in word_data]
boxes = [box['box'] for box in word_data]
word_labels = [label['label'] for label in word_data]

image_name = data['id']
image = Image.open(os.path.join('KR_ID_LAYOUT_TRAIN/images',image_name))


# encoding = processor(
#     image,
#     text=words,
#     boxes=boxes,
#     word_labels=word_labels,
#     padding="max_length",
#     truncation=True,
#     return_tensors="pt",
# )

text_encoding = processor.tokenizer(
    words,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

image_encoding = processor.image_processor(
    image,
    return_tensors="pt"
)

encoding = {
    **text_encoding,
    **image_encoding,
    "bbox": boxes,
    "labels": word_labels
}

print(encoding)