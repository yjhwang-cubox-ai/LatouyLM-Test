"""
클로드가 생성한 코드 + 내 데이터셋에 맞게 수정
"""

import os
import json
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LayoutLMv2ForTokenClassification,
    BertTokenizerFast,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

@dataclass
class Entity:
    text: str
    box: List[int]
    label: str

@dataclass
class DocumentExample:
    id: str
    document_class: str
    entities: List[Entity]
    image_path: str = None  # 이미지 경로는 나중에 설정

class KoreanDocumentDataset(Dataset):
    def __init__(
        self,
        examples: List[DocumentExample],
        tokenizer: BertTokenizerFast,
        max_length: int = 512,
        label_map: Dict[str, int] = None
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map or self._create_label_map()

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def _create_label_map(self) -> Dict[str, int]:
        # 모든 예제에서 고유한 레이블 수집
        unique_labels = set()
        for example in self.examples:
            for entity in example.entities:
                unique_labels.add(entity.label)
        # 레이블을 인덱스에 매핑
        return {label: idx for idx, label in enumerate(sorted(unique_labels))}

    def normalize_bbox(self, bbox: List[int], width: int, height: int) -> List[int]:
        return [
            int(1000 * bbox[0] / width),
            int(1000 * bbox[1] / height),
            int(1000 * bbox[2] / width),
            int(1000 * bbox[3] / height),
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # 이미지 로드 및 전처리
        image = Image.open(example.image_path).convert("RGB")
        width, height = image.size
        image_tensor = self.image_transform(image)

        # 엔티티들을 텍스트 순서대로 정렬 (top-to-bottom, left-to-right)
        sorted_entities = sorted(
            example.entities,
            key=lambda x: (x.box[1], x.box[0])  # y좌표 먼저, 그 다음 x좌표
        )

        words = []
        boxes = []
        labels = []

        for entity in sorted_entities:
            words.append(entity.text)
            boxes.append(entity.box)
            labels.append(self.label_map[entity.label])

        normalized_boxes = [self.normalize_bbox(box, width, height) for box in boxes]
        # print(f"Normalized boxes: {normalized_boxes}")

        # 토큰화
        tokenized = self.tokenizer(
            words,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors="pt"
        )

        # token to word index 매핑으로 bbox와 label 조정
        word_ids = tokenized.word_ids()

        bbox_inputs = []
        label_inputs = []

        for word_id in word_ids:
            if word_id is None:
                bbox_inputs.append([0, 0, 0, 0])
                label_inputs.append(-100)
            else:
                bbox_inputs.append(normalized_boxes[word_id])
                label_inputs.append(labels[word_id])

        for box in bbox_inputs:
            if not all(0 <= coord <= 1000 for coord in box):
                print(f"Warning: Invalid bbox coordinates: {box}")
                # 문제가 있는 좌표를 0~1000 범위로 클리핑
                box = [min(max(coord, 0), 1000) for coord in box]

        encoding = {
            "input_ids": tokenized["input_ids"].squeeze(0),            
            "token_type_ids": tokenized["token_type_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "bbox": torch.tensor(bbox_inputs),            
            "labels": torch.tensor(label_inputs),
            "image": image_tensor
        }

        return encoding

def load_dataset(data_dir: str) -> List[DocumentExample]:
    examples = []
    annotation_dir = os.path.join(data_dir, "annotations")
    image_dir = os.path.join(data_dir, "images")
    
    # JSON 파일들 로드
    for filename in os.listdir(annotation_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(annotation_dir, filename)
                
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Entity 객체들 생성
            entities = [
                Entity(
                    text=entity["text"],
                    box=entity["box"],
                    label=entity["label"]
                )
                for entity in data["entities"]
            ]
            
            # DocumentExample 객체 생성
            example = DocumentExample(
                id=data["id"],
                document_class=data["document_class"],
                entities=entities,
                image_path=os.path.join(image_dir, data['id'])
            )
            examples.append(example)
    
    return examples

def train_model(
    model: LayoutLMv2ForTokenClassification,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    device: torch.device,
    learning_rate: float = 2e-5,
) -> None:
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    model.to(device)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            print(f"loss: {loss}")
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        print("="*100)
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                
                logits = outputs.logits
                pred = torch.argmax(logits, dim=2)
                
                # Mask out padding tokens
                mask = batch["labels"] != -100
                predictions.extend(pred[mask].cpu().numpy())
                true_labels.extend(batch["labels"][mask].cpu().numpy())
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    DATA_DIR = "/data/KR_ID_LAYOUT_TRAIN_V2"
    TOKENIZER_NAME = "klue/bert-base"  # 한국어 BERT 토크나이저
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    MAX_LENGTH = 512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 로드
    examples = load_dataset(DATA_DIR)
    
    # 랜덤 분할
    train_examples, val_examples = train_test_split(
        examples,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # 토크나이저 로드
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_NAME)

    # 데이터셋 생성
    train_dataset = KoreanDocumentDataset(train_examples, tokenizer, MAX_LENGTH)
    val_dataset = KoreanDocumentDataset(
        val_examples, 
        tokenizer, 
        MAX_LENGTH,
        label_map=train_dataset.label_map  # 학습 데이터의 레이블 맵 재사용
    )

    # 레이블 개수 설정
    num_labels = len(train_dataset.label_map)
    
    # 모델 로드
    model = LayoutLMv2ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv2-base-uncased",
        num_labels=num_labels
    )
    model.resize_token_embeddings(len(tokenizer))

    # DataLoader 생성
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # 모델 학습
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=NUM_EPOCHS,
        device=DEVICE
    )

    # 모델과 레이블 맵 저장
    output_dir = "output!!!!!!!!!!"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 레이블 맵 저장
    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(train_dataset.label_map, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()