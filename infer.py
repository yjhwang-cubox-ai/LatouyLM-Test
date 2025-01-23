import torch
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import os
from datasets import load_dataset
from funsd_dataset import FUNSDDataset

def load_funsd_dataset(base_dir):
    # 이미지와 주석 파일이 있는 디렉토리
    dataset_dir = os.path.join(base_dir, "dataset")
    
    def load_split(split):
        images_dir = os.path.join(dataset_dir, split, "images")
        annotations_dir = os.path.join(dataset_dir, split, "annotations")
        
        data = {
            'image_paths': [],
            'words': [],
            'bboxes': [],
            'ner_tags': []
        }
        
        # 모든 주석 파일을 처리
        for filename in os.listdir(annotations_dir):
            if filename.endswith('.json'):
                # 주석 파일 로드
                with open(os.path.join(annotations_dir, filename), 'r', encoding='utf-8') as f:
                    annotation = json.load(f)
                
                # 관련 이미지 경로
                image_path = os.path.join(images_dir, filename.replace('.json', '.png'))
                
                # 각 form에서 데이터 추출
                words = []
                boxes = []
                labels = []
                
                for form in annotation['form']:
                    for word in form['words']:
                        words.append(word['text'])
                        boxes.append(word['box'])
                        # 레이블 매핑: O=0, question=1, answer=2, header=3
                        label_map = {'O': 0, 'question': 1, 'answer': 2, 'header': 3}
                        labels.append(label_map.get(form['label'], 0))
                
                data['image_paths'].append(image_path)
                data['words'].append(words)
                data['bboxes'].append(boxes)
                data['ner_tags'].append(labels)
        
        return data
    
    # 학습 및 테스트 데이터 로드
    train_data = load_split("training_data")
    test_data = load_split("testing_data")
    
    return {
        'train': train_data,
        'test': test_data
    }


class LayoutLMv2Predictor:
    def __init__(self, model_path):
        # 프로세서 및 모델 초기화
        self.processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
        self.model = LayoutLMv2ForTokenClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # FUNSD 데이터셋의 레이블 정의
        self.labels = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2color = {
            'OTHER': 'black',
            'HEADER': 'blue',
            'QUESTION': 'red',
            'ANSWER': 'green'
        }

    def preprocess_image(self, image, words, boxes):
        """이미지와 OCR 결과를 전처리"""
        # image = Image.open(image_path).convert("RGB")
        
        encoded_inputs = self.processor(
            images=image,
            text=words,
            boxes=boxes,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        for key in encoded_inputs.keys():
            encoded_inputs[key] = encoded_inputs[key].to(self.device)
            
        return image, encoded_inputs

    def predict(self, img, words, boxes):
        """이미지에서 예측 수행"""
        image, encoded_inputs = self.preprocess_image(img, words, boxes)
        
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
        
        predictions = outputs.logits.cpu().numpy()
        predictions = np.argmax(predictions, axis=-1)[0]
        
        # 특수 토큰 제외하고 예측 결과 매핑
        predicted_labels = []
        for prediction, box in zip(predictions[1:-1], boxes):  # CLS와 SEP 토큰 제외
            if prediction != -100:
                predicted_labels.append(self.id2label[prediction])
        
        return predicted_labels

    def visualize_predictions(self, image, words, boxes, predicted_labels):
        """예측 결과를 시각화"""
        # image = Image.open(image_path).convert("RGB")
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image, "RGBA")
        
        # 각 단어와 레이블에 대해 박스 그리기
        for word, box, label in zip(words, boxes, predicted_labels):
            # 레이블에서 B- 또는 I- 제거하고 타입 추출
            entity_type = label[2:] if label != 'O' else 'OTHER'
            color = self.label2color.get(entity_type, 'gray')
            
            # 박스 그리기 (반투명)
            draw.rectangle(box, outline=color, fill=(0, 0, 0, 32))
            
            # 텍스트 그리기
            draw.text((box[0], box[1]-20), f"{word} ({label})", fill=color)
        
        return image

    def process_document(self, img, words, boxes, output_path=None):
        """문서 처리 전체 파이프라인"""
        # 예측 수행
        predicted_labels = self.predict(img[0], words[0], boxes[0])
        
        # 결과 시각화
        visualized_image = self.visualize_predictions(
            img[0], words[0], boxes[0], predicted_labels
        )
        
        # 결과 저장
        if output_path:
            visualized_image.save(output_path)
            
            # JSON 형식으로 결과 저장
            results = {
                "words": words,
                "boxes": boxes,
                "predictions": predicted_labels
            }
            json_path = output_path.rsplit(".", 1)[0] + "_results.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        return visualized_image, predicted_labels

def main():
    # 모델 경로 설정
    model_path = "layoutlmv2-finetuned-funsd-v2/checkpoint-1000"
    
    # 예측기 초기화
    predictor = LayoutLMv2Predictor(model_path)

    # 데이터셋 로드
    funsd_data = load_funsd_dataset("/data/FUNSD")
    test_dataset = FUNSDDataset(funsd_data['train'], batch_size=4)

    output_dir = "output_predictions"
    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, data in enumerate(test_dataset):
        for i in range(len(data['images'])):
            sample = {
                'images': [data['images'][i]],
                'words': [data['words'][i]],
                'boxes': [data['boxes'][i]]
            }       

            output_path = os.path.join(output_dir, f"pred_{batch_idx}_{i}.jpg")

            visualized_image, predictions = predictor.process_document(
                sample['images'],
                sample['words'],
                sample['boxes'],
                output_path=output_path
            )

            print(f"Processed image {batch_idx}_{i}")

    print(f"All predictions saved to {output_dir}/")

if __name__ == "__main__":
    main()