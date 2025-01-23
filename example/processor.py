from transformers import LayoutLMv2Processor
from PIL import Image

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

image = Image.open(
    "test.png"
).convert("RGB")

encoding = processor(
    image, return_tensors="pt"
)
print(encoding.keys())