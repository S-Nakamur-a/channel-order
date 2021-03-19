from efficientnet_pytorch import EfficientNet


def get_model(model_name: str):
    return EfficientNet.from_pretrained(model_name, advprop=True, num_classes=1)
