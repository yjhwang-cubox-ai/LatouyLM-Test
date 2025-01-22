from transformers import LayoutLMv2Config, LayoutLMv2Model

configuration = LayoutLMv2Config()

model = LayoutLMv2Model(config=configuration)

configuration = model.config