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

    def _clean_text(self, text_list: list) -> str:
        """
        텍스트 리스트를 정제하여 하나의 문자열로 만듭니다.
        중복된 단어를 제거하고 적절히 공백을 추가합니다.
        """
        # 중복 제거하면서 순서 유지
        unique_words = []
        seen = set()
        for word in text_list:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)
        
        # 특별한 처리가 필요한 경우 (예: 주소, 날짜 등의 포맷)
        text = " ".join(unique_words)
        
        # 주소 형식 정제
        if "동" in text and "호" in text:
            text = text.replace(" 동", "동").replace(" 호", "호")
        elif "동" in text:
            text = text.replace(" 동", "동")
        
        # 날짜 형식 정제
        if "." in text:
            text = text.replace(" .", ".")
        
        return text.strip()
    
    def postprocess(self, logits, words, word_ids):
        # entity별 텍스트 저장
        results = {}
        prev_entity = None
        current_text = []
        seen_words = set()

        for word_idx, pred_idx in enumerate(logits):
            word_id = word_ids[word_idx]
            if word_id is not None:
                entity = self.index_to_label[pred_idx]
                if entity !="O":
                    # 새로운 엔티티 시작
                    if entity != prev_entity:
                        if prev_entity is not None:
                            # 이전 엔티티의 텍스트를 정제하여 저장
                            cleaned_text = self._clean_text(current_text)
                            if cleaned_text:
                                results[prev_entity] = cleaned_text
                        current_text = []
                        seen_words = set()  # 새 엔티티 시작시 seen_words 초기화

                    word = words[word_id]
                    if word not in seen_words:
                        current_text.append(word)
                        seen_words.add(word)
                    prev_entity = entity
        
        # 마지막 엔티티 처리
        if current_text and prev_entity is not None:
            results[prev_entity] = " ".join(current_text)
            
        return results


    def predict(self, image_paths, texts, bboxes):
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            texts = [texts]
            bboxes = [bboxes]
        
        batch_size = len(image_paths)
        all_sorted_texts, all_normalized_boxes, images = [], [], []

        for i in range(batch_size):
            sorted_texts, sorted_boxes = self._sort_by_reading_order(texts[i], bboxes[i])
            all_sorted_texts.append(sorted_texts)

            image = Image.open(image_paths[i]).convert('RGB')
            width, height = image.size
            normalized_boxes = [self.normalize_bbox(box, width, height) for box in sorted_boxes]
            all_normalized_boxes.append(normalized_boxes)

            resized = image.resize((224, 224), Image.LANCZOS)
            img_arr = np.array(resized).transpose(2,0,1).astype(np.float32)
            images.append(img_arr)
        
        encoding = self.tokenizer(
            all_sorted_texts,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            max_length=512
        )

        batch_bbox = []
        for i in range(batch_size):
            word_ids = encoding.word_ids(batch_index=i)
            sample_bbox = []
            for word_id in word_ids:
                if word_id is None:
                    sample_bbox.append([0, 0, 0, 0])
                else:
                    sample_bbox.append(all_normalized_boxes[i][word_id])
            # 좌표 클리핑
            sample_bbox = [[min(max(coord, 0), 1000) for coord in box] for box in sample_bbox]
            batch_bbox.append(sample_bbox)

        inputs = {
            "input_ids": encoding["input_ids"],
            "bbox": np.array(batch_bbox, dtype=np.int64),
            "image": np.stack(images, axis=0),
            "attention_mask": encoding["attention_mask"],
            "token_type_ids": encoding["token_type_ids"]
        }

        outputs = self.session.run(None, inputs)
        logits = outputs[0].argmax(-1)  # shape: (batch_size, seq_length)
        
        results = []
        for i in range(batch_size):
            word_ids = encoding.word_ids(batch_index=i)
            sample_result = self.postprocess(logits[i], all_sorted_texts[i], word_ids)
            results.append(sample_result)
        
        return results[0] if batch_size == 1 else results

def main():
    model = LayoutLMv2ONNX('layoutlmv2.onnx')

    # OCR 결과
    data_path = 'TESTSET'

    image_paths, words, boxes = [], [], []
    anno_path, img_path = os.path.join(data_path, 'annotations'), os.path.join(data_path, 'images')
    anno_files = os.listdir(anno_path)
    for anno_file in anno_files:
        with open(os.path.join(anno_path, anno_file), 'r', encoding='utf-8') as f:
            ocr_result = json.load(f)
        image_paths.append(os.path.join(img_path, ocr_result['id']))
        words.append([data['text'] for data in ocr_result['entities']])
        boxes.append([data['box'] for data in ocr_result['entities']])

    results = model.predict(image_paths=image_paths, texts=words, bboxes=boxes)

    print(results)

if __name__ == "__main__":
    main()