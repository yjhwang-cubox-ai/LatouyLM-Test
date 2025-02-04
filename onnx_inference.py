import os
import json
import numpy as np
import torch
import onnxruntime
from PIL import Image
from transformers import AutoTokenizer

class LayoutLMv2ONNX:
    def __init__(self, model_file):        
        self.tokenizer = AutoTokenizer.from_pretrained('tokenizer')
        self.onnx_model = model_file
        self.session = onnxruntime.InferenceSession(self.onnx_model)

        # 레이블 맵 로드
        with open(os.path.join('tokenizer', 'label_map.json'), 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        
        self.index_to_label = {v: k for k, v in self.label_map.items()}

        self.y_threshold = 10
    
    def _sort_by_reading_order(self, words: list, boxes: list) -> tuple:
        """
        바운딩 박스 좌표를 기준으로 읽기 순서대로 정렬합니다.
        y좌표 차이가 임계값보다 작으면 같은 줄로 간주하고 x좌표로 정렬합니다.
        """
        # (단어, 박스) 쌍 생성
        word_box_pairs = list(zip(words, boxes))
        
        # y좌표로 그룹화
        y_groups = {}
        for word, box in word_box_pairs:
            y_coord = box[1]  # y1 좌표
            assigned = False
            
            # 기존 그룹과 비교
            for group_y in sorted(y_groups.keys()):
                if abs(y_coord - group_y) <= self.y_threshold:
                    y_groups[group_y].append((word, box))
                    assigned = True
                    break
            
            # 새로운 그룹 생성
            if not assigned:
                y_groups[y_coord] = [(word, box)]
        
        # 각 그룹 내에서 x좌표로 정렬하고, 그룹은 y좌표로 정렬
        sorted_words = []
        sorted_boxes = []
        
        for y_coord in sorted(y_groups.keys()):
            # x좌표로 그룹 내 정렬
            group = sorted(y_groups[y_coord], key=lambda x: x[1][0])  # x1 좌표로 정렬
            
            # 정렬된 결과 추가
            group_words, group_boxes = zip(*group)
            sorted_words.extend(group_words)
            sorted_boxes.extend(group_boxes)
        
        return sorted_words, sorted_boxes
    
    def normalize_bbox(self, bbox: list, width: int, height: int) -> list:
        return[
            int(1000 * bbox[0] / width),
            int(1000 * bbox[1] / height),
            int(1000 * bbox[2] / width),
            int(1000 * bbox[3] / height),
        ]
    
    def predict(self, image_path, texts, bboxes):
        texts, bboxes = self._sort_by_reading_order(texts, bboxes)        
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        # 텍스트 처리
        encoding = self.tokenizer(
            texts,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            max_length=512
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        token_type_ids = encoding["token_type_ids"]

        batch_size, seq_length = input_ids.shape

        word_ids = encoding.word_ids()

        # bbox 처리
        normalized_boxes = [self.normalize_bbox(box, width, height) for box in bboxes]

        bbox_inputs = []
        for word_id in word_ids:
            if word_id is None:
                bbox_inputs.append([0, 0, 0, 0])
            else:
                bbox_inputs.append(normalized_boxes[word_id])
        
        for box in bbox_inputs:
            if not all(0 <= coord <= 1000 for coord in box):
                print(f"Warning: Invalid bbox coordinates: {box}")
                # 문제가 있는 좌표를 0~1000 범위로 클리핑
                box = [min(max(coord, 0), 1000) for coord in box]
        
        bboxes = np.expand_dims(np.array(bbox_inputs), axis=0)
        resized_image = np.array(image.resize((224, 224), Image.LANCZOS))

        inputs = {
            "input_ids": input_ids,
            "bbox": bboxes,
            "image": resized_image,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }

        outputs = self.session.run(None, inputs)
        print(outputs.shape)

def main():
    model = LayoutLMv2ONNX('layoutlmv2.onnx')

    # OCR 결과
    data_path = 'TESTSET'
    with open(os.path.join(data_path, '00003132.json'), 'r', encoding='utf-8') as f:
        ocr_result = json.load(f)
    
    image_path = os.path.join(data_path, ocr_result['id'])
    words = [data['text'] for data in ocr_result['entities']]
    boxes = [data['box'] for data in ocr_result['entities']]

    results = model.predict(image_path=image_path, texts=words, bboxes=boxes)

if __name__ == "__main__":
    main()