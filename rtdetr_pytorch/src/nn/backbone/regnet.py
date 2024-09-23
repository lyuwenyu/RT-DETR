import torch
import torch.nn as nn 
from transformers import RegNetModel


from src.core import register

__all__ = ['RegNet']

@register
class RegNet(nn.Module):
    def __init__(self, configuration, return_idx=[0, 1, 2, 3]):
        super(RegNet, self).__init__()  
        self.model = RegNetModel.from_pretrained("facebook/regnet-y-040")
        self.return_idx = return_idx


    def forward(self, x):
        
        outputs = self.model(x, output_hidden_states = True)
        x = outputs.hidden_states[2:5]

        return x