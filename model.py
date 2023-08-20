import torch
import torch.nn as nn
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

class PointPoseNet(nn.Module):
    '''
    As main model we will use pretrained network from torchvision in base of Faster-RCNN model : KeypointRCNN.
    The weights will be downloading during the execution of inference

    '''
    def __init__(self, postprocesser=None, device='cpu') -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT, progress=True)
        self.model.to(self.device)
        self.model.eval()
        self.postprocesser = postprocesser
        

    def forward(self, image, task):
        shape = image[0].shape[1]
        out = self.model(image)
        print(f'in model out: {len(out)}')
        result = self.postprocesser.process(out, task)
        result = [r / shape for r in result]
        return result
    
