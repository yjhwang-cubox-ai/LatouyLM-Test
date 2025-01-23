from datasets import load_dataset
from PIL import Image, ImageDraw
from transformers import LayoutLMv2Processor
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from torch.utils.data import DataLoader
from transformers import LayoutLMv2ForTokenClassification, TrainingArguments, Trainer
from evaluate import load
import numpy as np


def visualize_predictions(image, words, boxes, predicted_labels):
        """예측 결과를 시각화"""
        # image = Image.open(image_path).convert("RGB")
        image = Image.fromarray(image)
        width, height = image.size
        draw = ImageDraw.Draw(image, "RGBA")

        label2color = {
            'OTHER': 'black',
            'HEADER': 'blue',
            'QUESTION': 'red',
            'ANSWER': 'green'
        }
        
        # 각 단어와 레이블에 대해 박스 그리기
        for word, box, label in zip(words, boxes, predicted_labels):
            # 레이블에서 B- 또는 I- 제거하고 타입 추출
            entity_type = label[2:] if label != 'O' else 'OTHER'
            color = label2color.get(entity_type, 'gray')

             # 좌표 스케일 복원
            scaled_box = [
                box[0] * width / 1000,
                box[1] * height / 1000,
                box[2] * width / 1000,
                box[3] * height / 1000
            ]
            
            # 박스 그리기 (반투명)
            draw.rectangle(scaled_box, outline=color, fill=(0, 0, 0, 32))
            
            # 텍스트 그리기
            draw.text((box[0], box[1]-20), f"{word} ({label})", fill=color)
        
        return image

datasets = load_dataset("nielsr/funsd", split="train", trust_remote_code=True)
labels = datasets.features['ner_tags'].feature.names
id2label = {v: k for v, k in enumerate(labels)}

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
model = LayoutLMv2ForTokenClassification.from_pretrained("layoutlmv2-finetuned-funsd-v2-0123/checkpoint-2000", num_labels=len(labels))


data = datasets[0]
image = Image.open(data['image_path']).convert('RGB')
words = data['words']
boxes = data['bboxes']
word_labels = data["ner_tags"]
encoding = processor(
    image,
    words,
    boxes=boxes,
    word_labels=word_labels,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

outputs = model(**encoding)
logits, loss = outputs.logits, outputs.loss
predicted_token_class_ids = logits.argmax(-1)
predicted_tokens_classes = [id2label[t.item()] for t in predicted_token_class_ids[0]]

visualized_image = visualize_predictions(image=np.array(image), words=words, boxes=boxes, predicted_labels=predicted_tokens_classes)
img_array = np.array(visualized_image)
visualized_image.save('result_0123.jpg')