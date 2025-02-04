import torch
from transformers import AutoTokenizer, LayoutLMv2ForTokenClassification

# 1. 파인튜닝된 모델과 토크나이저 로드 (모델명 또는 경로를 수정하세요)
model_name = "model_0204"  # 예: "username/layoutlmv2-finetuned"
model = LayoutLMv2ForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 모델이 출력으로 딕셔너리가 아닌 튜플을 반환하도록 설정합니다.
model.config.return_dict = False

# 2. 더미 입력 생성
# 예시 문장을 이용해 토큰화
dummy_text = ["주민등록증"]
encoding = tokenizer(
    dummy_text,
    padding="max_length",
    truncation=True,
    max_length=512,
    is_split_into_words=True,
    return_tensors='pt'
)
# encoding = tokenizer(dummy_text, return_tensors="pt")
input_ids = encoding["input_ids"]        # [batch_size, seq_length]
attention_mask = encoding["attention_mask"]

batch_size, seq_length = input_ids.shape

# bbox: [batch_size, seq_length, 4]
# 여기서는 각 토큰마다 [0, 0, 1000, 1000] 값을 부여합니다.
dummy_bbox = torch.tensor([[[0, 0, 1000, 1000]]]).repeat(batch_size, seq_length, 1)

# image: [batch_size, 3, 224, 224]
# 일반적인 LayoutLMv2의 비전 모듈 입력 크기를 사용 (필요에 따라 변경)
dummy_image = torch.randn(batch_size, 3, 224, 224)

# token_type_ids: [batch_size, seq_length]
dummy_token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)

# 3. 모델을 평가 모드로 전환
model.eval()

# 4. ONNX 모델로 내보내기
onnx_model_path = "layoutlmv2.onnx"

torch.onnx.export(
    model,
    (input_ids, dummy_bbox, dummy_image, attention_mask, dummy_token_type_ids),  # 모델의 입력 순서에 맞게 튜플 구성
    onnx_model_path,
    input_names=["input_ids", "bbox", "image", "attention_mask", "token_type_ids"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "bbox": {0: "batch_size", 1: "seq_length"},
        "image": {0: "batch_size"},
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        "token_type_ids": {0: "batch_size", 1: "seq_length"},
        "output": {0: "batch_size"}
    },
    opset_version=12,           # 필요에 따라 최신 opset 버전 사용 가능
    do_constant_folding=True    # 상수 폴딩 최적화 적용
)

print(f"ONNX 모델이 성공적으로 '{onnx_model_path}'로 내보내졌습니다.")