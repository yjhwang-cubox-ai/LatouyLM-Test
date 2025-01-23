from datasets import load_dataset
from PIL import Image
from transformers import LayoutLMv2Processor
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from torch.utils.data import DataLoader
from transformers import LayoutLMv2ForTokenClassification, TrainingArguments, Trainer
from evaluate import load
import numpy as np
import torch

# 1. 데이터셋과 레이블 준비
datasets = load_dataset("nielsr/funsd", trust_remote_code=True)
labels = datasets['train'].features['ner_tags'].feature.names

id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}

# 2. 프로세서 및 특징 정의
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(ClassLabel(names=labels)),
})

# 3. 전처리 함수
def preprocess_data(examples):
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    words = examples['words']
    boxes = examples['bboxes']
    word_labels = examples['ner_tags']

    encoded_inputs = processor(
        images=images, 
        text=words, 
        boxes=boxes, 
        word_labels=word_labels,
        padding="max_length",
        truncation=True
    )
    
    return encoded_inputs

# 4. 테스트 데이터 준비
test_dataset = datasets['train'].map(
    preprocess_data,
    batched=True,
    remove_columns=datasets['train'].column_names,
    features=features
)
test_dataset.set_format(type="torch")
test_dataloader = DataLoader(test_dataset, batch_size=2)

# 5. 모델 불러오기
model_path = "layoutlmv2-finetuned-funsd-v2-0123/checkpoint-2000"  # 체크포인트 경로
model = LayoutLMv2ForTokenClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 6. 평가 함수 정의
def compute_metrics(predictions, labels):
    predictions = np.argmax(predictions, axis=2)
    
    metric = load("seqeval")
    
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return results

# 7. 테스트 실행
def test_model():
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # 데이터를 GPU로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            bbox = batch['bbox'].to(device)
            image = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            # 모델 예측
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                bbox=bbox,
                image=image
            )
            
            predictions = outputs.logits.cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
    # 최종 평가
    results = compute_metrics(np.array(all_predictions), np.array(all_labels))
    return results

if __name__ == "__main__":
    # 테스트 실행
    results = test_model()
    
    # 결과 출력
    print("\nTest Results:")
    print(f"Overall Precision: {results['overall_precision']:.4f}")
    print(f"Overall Recall: {results['overall_recall']:.4f}")
    print(f"Overall F1: {results['overall_f1']:.4f}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    
    # 각 엔티티별 결과 출력
    print("\nEntity-level Results:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for metric_name, score in value.items():
                print(f"  {metric_name}: {score:.4f}")