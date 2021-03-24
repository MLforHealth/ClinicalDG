import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
           

class EmbModel(nn.Module):
    def __init__(self, emb_type, pretrain, concat_features = 0):
        super().__init__()
        self.emb_type = emb_type
        self.pretrain = pretrain
        self.concat_features = concat_features
        
        if emb_type == 'densenet':
            model = models.densenet121(pretrained=pretrain)
            self.encoder = nn.Sequential(*list(model.children())[:-1]) #https://discuss.pytorch.org/t/densenet-transfer-learning/7776/2
            self.emb_dim = model.classifier.in_features
        elif emb_type == 'resnet':
            model = models.resnet50(pretrained=pretrain)
            self.encoder = nn.Sequential(*list(model.children())[:-1])
            self.emb_dim = list(model.children())[-1].in_features
            
        self.n_outputs = self.emb_dim + concat_features      
        
    def forward(self, inp):
        if isinstance(inp, dict): # dict with image and additional feature(s) to concat to embedding
            x = inp['img']
            concat = inp['concat']
            assert(concat.shape[-1] == self.concat_features)
        else: # tensor image
            assert(self.concat_features == 0)
            x = inp
        
        x = self.encoder(x).squeeze(-1).squeeze(-1)
        if self.emb_type == 'densenet':
            x = F.relu(x)
            x = F.avg_pool2d(x, kernel_size = 7).view(x.size(0), -1)
        
        if isinstance(inp, dict):
            x = torch.cat([x, concat], dim = -1)
        
        return x 
   
       