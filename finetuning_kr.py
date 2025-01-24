from datasets import load_dataset
from PIL import Image
from transformers import LayoutLMv2Processor
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from torch.utils.data import DataLoader
from transformers import LayoutLMv2ForTokenClassification, TrainingArguments, Trainer, BertTokenizerFast, LayoutLMv2FeatureExtractor
from evaluate import load
import evaluate
import numpy as np
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):    
    def __init__(self, directory, processor):
        self.image_dir = os.path.join(directory, 'images')
        self.annotation_dir = os.path.join(directory, 'annotations')
        self.annotation_files = sorted(os.listdir(self.annotation_dir))
        self.processor = processor
        self.features = Features({
            'image': Array3D(dtype="int64", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'token_type_ids': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': Sequence(ClassLabel(names=labels))
        })
        self.labels = ['head', 'name', 'idnumber', 'address', '발급일자', '발급기관']
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.cached_data = []        
        
        for ann_file in self.annotation_files:
            with open(os.path.join(self.annotation_dir, ann_file), 'r', encoding='utf-8') as f:
                self.cached_data.append(json.load(f))

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        annotation = self.cached_data[idx]
        texts, bboxes, label_ids = zip(*[
           (data['text'], data['box'], self.label2id[data['label']])
           for data in annotation["entities"]
       ])
        
        image = Image.open(os.path.join(self.image_dir, annotation['id'])).convert("RGB")

        encoded_inputs = self.processor(
            images=image,
            text=list(texts), 
            boxes=list(bboxes),
            word_labels=list(label_ids),
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Features에 맞게 데이터 형식 조정
        return {k: v.squeeze(0) for k, v in encoded_inputs.items()}


#Prepare the data
# datasets = load_dataset("nielsr/funsd", trust_remote_code=True)

# Preprocess data
# labels = datasets['train'].features['ner_tags'].feature.names
labels = ['head', 'name', 'idnumber', 'address', '발급일자', '발급기관']
# print(labels)

id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}
# print(label2id)

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

datasets = CustomDataset('KR_ID_LAYOUT_TRAIN', processor=processor)

# datasets.set_format(type="torch")
print(datasets.features.keys())


# # KOREAN
# tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
# feature_extractor = LayoutLMv2FeatureExtractor()
# processor = LayoutLMv2Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)


# we need to define custom features
# features = Features({
#     'image': Array3D(dtype="int64", shape=(3, 224, 224)),
#     'input_ids': Sequence(feature=Value(dtype='int64')),
#     'attention_mask': Sequence(Value(dtype='int64')),
#     'token_type_ids': Sequence(Value(dtype='int64')),
#     'bbox': Array2D(dtype="int64", shape=(512, 4)),
#     'labels': Sequence(ClassLabel(names=labels)),
# })

# def preprocess_data(examples):
#     images = [Image.open(path).convert("RGB") for path in examples['image_path']]
#     words = examples['words']
#     boxes = examples['bboxes']
#     word_labels = examples['ner_tags']

#     encoded_inputs = processor(images=images, text=words, boxes=boxes, word_labels=word_labels, padding="max_length", truncation=True)

#     return encoded_inputs


# #origin
# # train_dataset = datasets['train'].map(preprocess_data, batched=True, remove_columns=datasets['train'].column_names, features=features)
# # test_dataset = datasets['test'].map(preprocess_data, batched=True, remove_columns=datasets['test'].column_names, features=features)
# train_dataset = datasets.map(preprocess_data, batch=True)

# train_dataset.set_format(type="torch")
# # test_dataset.set_format(type="torch")

# print(train_dataset.features.keys())

train_dataloader = DataLoader(datasets, batch_size=4, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=2)

batch = next(iter(train_dataloader))

model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                                      num_labels=len(label2id))

model.config.id2label = id2label
model.config.label2id = label2id


return_entity_level_metrics = True

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # seqeval metric 로드
    metric = load("seqeval")

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # seqeval을 사용하여 metrics 계산
    results = metric.compute(predictions=true_predictions, references=true_labels)
    
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

class FunsdTrainer(Trainer):
    def get_train_dataloader(self):
      return train_dataloader

    # def get_test_dataloader(self, test_dataset):
    #   return test_dataloader

args = TrainingArguments(
    output_dir="layoutlmv2-finetuned-funsd-v2-0124", # name of directory to store the checkpoints
    max_steps=2000, # we train for a maximum of 1,000 batches
    warmup_ratio=0.1, # we warmup a bit
    fp16=True, # we use mixed precision (less memory consumption)
    push_to_hub=False # after training, we'd like to push our model to the hub
)

trainer = FunsdTrainer(
    model=model,
    args=args,
    # compute_metrics=compute_metrics
)

trainer.train()