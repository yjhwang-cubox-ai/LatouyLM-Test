#sequence classification: 주어진 입력 문서(sequence)를 하나의 클래스(라벨)로 분류하는 작업, 텍스트와 시각적 정보를 모두 활용하여 문서의 유형을 예측

from transformers import AutoProcessor, LayoutLMv2ForSequenceClassification, set_seed
from PIL import Image
import torch
from datasets import load_dataset

set_seed(0)

dataset = load_dataset("aharley/rvl_cdip", split="train", streaming=True, trust_remote_code=True)

iterator = iter(dataset)
for _ in range(2):
    data = next(iterator)

image = data["image"].convert("RGB")
image.save('test4.jpg')

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", num_labels=dataset.info.features["label"].num_classes)

encoding = processor(image, return_tensors="pt")
sequence_label = torch.tensor([data['label']])

outputs = model(**encoding, labels=sequence_label)

loss, logits = outputs.loss, outputs.logits
predicted_idx = logits.argmax(dim=-1).item()
predicted_answer = dataset.info.features["label"].names[4]

print(f"predicted_idx: {predicted_idx}")
print(f"predicted_answer: {predicted_answer}")
