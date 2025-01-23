from transformers import AutoProcessor, LayoutLMv2Model, set_seed, LayoutLMv2Processor
from PIL import Image
import torch
from datasets import load_dataset

set_seed(0)

# processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")

dataset = load_dataset("hf-internal-testing/fixtures_docvqa", trust_remote_code=True)
# image_path = dataset["test"][0]["file"]
image_path = 'testing.png'
image = Image.open(image_path).convert("RGB")

encoding = processor(image, return_tensors="pt")

outputs = model(**encoding)
last_hidden_states = outputs.last_hidden_state

print("OUTPUT:")
print(outputs)
print("-------------")
print(last_hidden_states.shape)