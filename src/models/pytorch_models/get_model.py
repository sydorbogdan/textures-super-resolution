from src.models.pytorch_models.UNet.unet_model import UNet


def get_model(model_name: str = 'UNet', **kwargs):
    if model_name == 'UNet':
        return UNet(kwargs)
