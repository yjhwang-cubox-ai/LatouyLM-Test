import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LayoutLMv2ForTokenClassification,
    LayoutLMv2FeatureExtractor,
    LayoutLMv2TokenizerFast,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import pytesseract
from typing import Dict, List, Tuple
import json

class DocumentDataset(Dataset):
    def __init__(self, dataset_path: str, tokenizer, feature_extractor):
        """한글 문서 데이터셋 클래스 초기화
        
        Args:
            dataset_path: 데이터셋 경로
            tokenizer: LayoutLMv2 토크나이저
            feature_extractor: LayoutLMv2 특성 추출기
        """
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.images, self.labels = self._load_dataset()
        
    def _load_dataset(self) -> Tuple[List[Image.Image], List[Dict]]:
        """데이터셋 로드 및 전처리
        
        Returns:
            images: 전처리된 이미지 리스트
            labels: 레이블 정보 리스트
        """
        images = []
        labels = []
        # 데이터셋 디렉토리에서 이미지와 레이블 파일 로드
        for filename in os.listdir(os.path.join(self.dataset_path, 'images')):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # 이미지 로드
                image_path = os.path.join(self.dataset_path, 'images', filename)
                image = Image.open(image_path).convert('RGB')
                images.append(image)
                
                # 레이블 로드 (json 또는 다른 형식)
                label_path = os.path.join(
                    self.dataset_path, 
                    'labels', 
                    filename.rsplit('.', 1)[0] + '.json'
                )
                with open(label_path, 'r', encoding='utf-8') as f:
                    label = json.load(f)
                labels.append(label)
                
        return images, labels
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """데이터셋에서 단일 항목 추출 및 전처리
        
        Args:
            idx: 데이터 인덱스
            
        Returns:
            전처리된 특성과 레이블을 포함하는 딕셔너리
        """
        image = self.images[idx]
        label = self.labels[idx]
        
        # OCR 수행하여 텍스트와 바운딩 박스 추출
        ocr_result = pytesseract.image_to_data(
            image, 
            lang='kor', 
            output_type=pytesseract.Output.DICT
        )
        
        # 텍스트와 바운딩 박스 정보 추출
        words = []
        boxes = []
        for i in range(len(ocr_result['text'])):
            if ocr_result['text'][i].strip():
                words.append(ocr_result['text'][i])
                boxes.append([
                    ocr_result['left'][i],
                    ocr_result['top'][i],
                    ocr_result['left'][i] + ocr_result['width'][i],
                    ocr_result['top'][i] + ocr_result['height'][i]
                ])
        
        # 이미지 인코딩
        encoding = self.feature_extractor(
            image,
            return_tensors="pt"
        )
        
        # 텍스트 토큰화
        word_labels = self._align_labels(words, label)
        tokenized = self.tokenizer(
            words,
            boxes=boxes,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors="pt"
        )
        
        return {
            'input_ids': tokenized.input_ids.squeeze(),
            'attention_mask': tokenized.attention_mask.squeeze(),
            'bbox': tokenized.bbox.squeeze(),
            'pixel_values': encoding.pixel_values.squeeze(),
            'labels': torch.tensor(word_labels)
        }
    
    def _align_labels(self, words: List[str], label: Dict) -> List[int]:
        """OCR 결과와 레이블 정보 정렬
        
        Args:
            words: OCR로 추출한 단어 리스트
            label: 원본 레이블 정보
            
        Returns:
            정렬된 레이블 리스트
        """
        # 여기에 레이블 정렬 로직 구현
        # 예: BIO 태깅 방식으로 레이블 할당
        return [0] * len(words)  # 임시 더미 레이블

def train_layoutlmv2():
    """LayoutLMv2 모델 학습 함수"""
    # 토크나이저와 특성 추출기 초기화
    tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
    feature_extractor = LayoutLMv2FeatureExtractor()
    
    # 데이터셋 준비
    train_dataset = DocumentDataset(
        "path/to/train/dataset",
        tokenizer,
        feature_extractor
    )
    eval_dataset = DocumentDataset(
        "path/to/eval/dataset",
        tokenizer,
        feature_extractor
    )
    
    # 모델 초기화
    model = LayoutLMv2ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv2-base-uncased",
        num_labels=len(train_dataset.label_list)
    )
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir="./layoutlmv2-korean-documents",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    # 트레이너 초기화 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # 모델 학습
    trainer.train()
    
    # 모델 저장
    trainer.save_model("./layoutlmv2-korean-documents-final")

if __name__ == "__main__":
    train_layoutlmv2()