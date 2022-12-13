import torchvision
from torch import nn 
import torch

class wide_Resnet:
    def __init__(self , nums_class) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.num_class = nums_class

    def Resnet_50(self):
        model = torchvision.models.efficientnet_b0(pretrained=True)

        for param in model.parameters():
            param.required_grad = False


        num_ftrt = model.fc.in_features

        model.fc = nn.Linear(num_ftrt,self.num_class)
        model.to(self.device)

        return model
