import torch
import torch.nn as nn
from transformers import RegNetModel


__all__ = ['RegNet']

class RegNet(nn.Module):
    def __init__(self, variant="regnet-y-040", return_idx=[0, 1, 2, 3]):
        super(RegNet, self).__init__()
        
        self.variant_dict = {
            "regnet-y-040": "facebook/regnet-y-040",
            "regnet-y-080": "facebook/regnet-y-080",
            "regnet-y-160": "facebook/regnet-y-160",
            "regnet-y-320": "facebook/regnet-y-320",
        }

        if variant not in self.variant_dict:
            raise ValueError(f"Variant '{variant}' is not a valid RegNet model. Please choose from: {list(self.variant_dict.keys())}")
        
        self.model = RegNetModel.from_pretrained(self.variant_dict[variant])
        
        self.return_idx = return_idx

    def forward(self, x):
        outputs = self.model(x, output_hidden_states=True)
        
        x = outputs.hidden_states[self.return_idx[0]:self.return_idx[1]]
        
        return x
