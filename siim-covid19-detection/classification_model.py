from torch import nn
import timm


class Model(nn.Module):
    def __init__(self, num_classes=4, model_name="tf_efficientnet_l2_ns", pretrained=True):
        super(Model, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=3)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
