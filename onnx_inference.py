import numpy as np
import torch
import onnxruntime
from transformers import AutoTokenizer

# 1. onnx 모델 로드
onnx_model_path = 'layoutlmv2.onnx'
session = onnxruntime.InferenceSession(onnx_model_path)

# 2. 토크나이저 로드
model_name = 'model_0204'
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "This is a sample text for ONNX inference."
encoding = tokenizer(text, return_tensors="pt")
input_ids = encoding["input_ids"]         # shape: [batch_size, seq_length]
attention_mask = encoding["attention_mask"]

batch_size, seq_length = input_ids.shape

# 4. dummy bbox 생성
# 각 토큰에 대해 [0, 0, 1000, 1000] 값을 부여 (실제 bbox 좌표 필요시 수정)
dummy_bbox = torch.tensor([[[0, 0, 1000, 1000]]]).repeat(batch_size, seq_length, 1)

# 5. dummy image 생성
# LayoutLMv2의 비전 입력 크기 (3, 224, 224)를 따름
dummy_image = torch.randn(batch_size, 3, 224, 224)

# 6. dummy token_type_ids 생성 (기본값 0)
dummy_token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)

# 7. ONNX 런타임에 넣기 위해 numpy 배열로 변환
inputs = {
    "input_ids": input_ids.cpu().numpy(),
    "bbox": dummy_bbox.cpu().numpy(),
    "image": dummy_image.cpu().numpy(),
    "attention_mask": attention_mask.cpu().numpy(),
    "token_type_ids": dummy_token_type_ids.cpu().numpy(),
}

# 8. ONNX 모델로 추론 실행
outputs = session.run(None, inputs)

# outputs[0]에 logits가 반환되며 shape는 [batch_size, seq_length, num_labels] 입니다.
logits = outputs[0]
print("Logits shape:", logits.shape)
print("Logits:", logits)

# 9. (선택 사항) 토큰 분류의 경우, 각 토큰별로 가장 높은 점수를 가진 라벨 ID를 예측할 수 있습니다.
predicted_label_ids = np.argmax(logits, axis=-1)
print("Predicted Label IDs:", predicted_label_ids)