import json
import os
import numpy as np
from PIL import Image

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

class FUNSDDataset:
    def __init__(self, data, batch_size=2):
        self.image_paths = data['image_paths']
        self.words = data['words']
        self.bboxes = data['bboxes']
        self.ner_tags = data['ner_tags']
        self.batch_size = batch_size
        self.current_idx = 0
        
    def __len__(self):
        return len(self.image_paths)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_idx >= len(self):
            self.current_idx = 0
            raise StopIteration
            
        batch_images = []
        batch_words = []
        batch_boxes = []
        batch_labels = []
        
        for i in range(self.batch_size):
            if self.current_idx + i >= len(self):
                break
                
            # 이미지 로드 및 전처리
            img = Image.open(self.image_paths[self.current_idx + i]).convert("RGB")
            img = np.array(img)
            
            batch_images.append(img)
            batch_words.append(self.words[self.current_idx + i])
            batch_boxes.append(self.bboxes[self.current_idx + i])
            batch_labels.append(self.ner_tags[self.current_idx + i])
        
        self.current_idx += self.batch_size
        
        return {
            'images': batch_images,
            'words': batch_words,
            'boxes': batch_boxes,
            'labels': batch_labels
        }

# 사용 예시:

# # 데이터셋 로드
# funsd_data = load_funsd_dataset("/data/FUNSD")

# # 테스트 데이터셋 생성
# test_dataset = FUNSDDataset(funsd_data['test'], batch_size=1)

# # 데이터 순회
# for batch in test_dataset:
#     images = batch['images']
#     words = batch['words']
#     boxes = batch['boxes']
#     labels = batch['labels']
#     # 처리 로직...
#     print(f'images:\n{images}')
#     print(f'words:\n{words}')
#     print(f'boxes:\n{boxes}')
#     print(f'labels:\n{labels}')

# class FUNSD:
#     def __init__(self, base_dir):
#         self.base_dir = base_dir
#         self.train_data = self.load_funsd_dataset['train']
#         self.test_data = self.load_funsd_dataset['test']

#     def load_funsd_dataset(self):
#         dataset_dir = os.path.join(self.base_dir, "dataset")
        
#         def load_split(split):
#             images_dir = os.path.join(dataset_dir, split, "images")
#             annotations_dir = os.path.join(dataset_dir, split, "annotations")
            
#             data = {
#                 'image_paths': [],
#                 'words': [],
#                 'bboxes': [],
#                 'ner_tags': []
#             }
            
#             # 모든 주석 파일을 처리
#             for filename in os.listdir(annotations_dir):
#                 if filename.endswith('.json'):
#                     # 주석 파일 로드
#                     with open(os.path.join(annotations_dir, filename), 'r', encoding='utf-8') as f:
#                         annotation = json.load(f)
                    
#                     # 관련 이미지 경로
#                     image_path = os.path.join(images_dir, filename.replace('.json', '.png'))
                    
#                     # 각 form에서 데이터 추출
#                     words = []
#                     boxes = []
#                     labels = []
                    
#                     for form in annotation['form']:
#                         for word in form['words']:
#                             words.append(word['text'])
#                             boxes.append(word['box'])
#                             # 레이블 매핑: O=0, question=1, answer=2, header=3
#                             label_map = {'O': 0, 'question': 1, 'answer': 2, 'header': 3}
#                             labels.append(label_map.get(form['label'], 0))
                    
#                     data['image_paths'].append(image_path)
#                     data['words'].append(words)
#                     data['bboxes'].append(boxes)
#                     data['ner_tags'].append(labels)
            
#             return data
        
#         # 학습 및 테스트 데이터 로드
#         train_data = load_split("training_data")
#         test_data = load_split("testing_data")
        
#         return {
#             'train': train_data,
#             'test': test_data
#         }