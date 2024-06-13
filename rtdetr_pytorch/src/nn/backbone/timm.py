from src.core import register
import timm
import torch.nn as nn

__all__ = ['Timm']

@register
class Timm(nn.Module):
    def __init__(
            self,
            model_type : str = 'mobilenetv3_small_050.lamb_in1k'
        ):
        super().__init__()
        self.model = timm.create_model(model_type, features_only=True, pretrained=True)
        

    def forward(self, x):
        o = self.model(x)

        outputs = []

        for x in o[2:]:
            outputs.append(x)

        return outputs