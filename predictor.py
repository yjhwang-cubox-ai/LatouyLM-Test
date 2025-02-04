import os 
import json
import torch
from PIL import Image
from transformers import LayoutLMv2ForTokenClassification, AutoTokenizer
from torchvision import transforms

class DocumentPredictor:
    def __init__(self, model_path: str):
        self.model = LayoutLMv2ForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 레이블 맵 로드
        with open(os.path.join(model_path, 'label_map.json'), 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        
        # label_map 을 역매핑 (index -> label)
        self.index_to_label = {v: k for k, v in self.label_map.items()}

        # 이미지 전처리
        self.image_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        # GPU 사용 가능시 GPU 사용
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.y_threshold = 10
    
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
    
    def predict(self, image_path: str, words: list, boxes: list) -> dict:
        """
        이미지와 OCR 결과를 받아 entity를 예측한다.
        """        
        words, boxes = self._sort_by_reading_order(words, boxes)

        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        image_tensor = self.image_transform(image)

        # bbox 정규화
        normalized_boxes = [self.normalize_bbox(box, width, height) for box in boxes]

        # 토큰화
        tokenized = self.tokenizer(
            words,
            padding="max_length",
            truncation=True,
            max_length=512,
            is_split_into_words=True,
            return_tensors='pt'
        )

        word_ids = tokenized.word_ids()

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


        encoding = {
            "input_ids": tokenized["input_ids"],
            "token_type_ids": tokenized["token_type_ids"],
            "attention_mask": tokenized["attention_mask"],
            "bbox": torch.tensor([bbox_inputs]),
            "image": image_tensor.unsqueeze(0)
        }

        # 나머지 텐서들도 device 로 이동
        encoding = {k: v.to(self.device) for k,v in encoding.items()}

        # Prediction
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().cpu().numpy()

        # entity별 텍스트 저장
        results = {}
        prev_entity = None
        current_text = []
        seen_words = set()

        for word_idx, pred_idx in enumerate(predictions):
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




def main():
    # 모델 초기화
    predictor = DocumentPredictor(model_path='model_0204')

    # OCR 결과
    data_path = 'TESTSET'

    with open(os.path.join(data_path, '00003132.json'), 'r', encoding='utf-8') as f:
        ocr_result = json.load(f)
    
    image_path = ocr_result['id']
    words = [data['text'] for data in ocr_result['entities']]
    boxes = [data['box'] for data in ocr_result['entities']]

    # 예측
    results = predictor.predict(image_path=os.path.join(data_path, image_path), words=words, boxes=boxes)
    
    # 결과 출력
    print("예측 결과:")
    for entity_type, text in results.items():
        print(f"{entity_type}: {text}")


if __name__ == "__main__":
    main()