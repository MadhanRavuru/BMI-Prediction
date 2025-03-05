import torch
from torch import nn
import torchvision.models as models


class BMIPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load MobileNetV3 large as the feature extractor
        self.base_model = models.mobilenet_v3_large(pretrained=True) 
        
        # Freeze all layers
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last 6 layers of the feature extractor for fine-tuning
        for param in self.base_model.features[-6:].parameters():
            param.requires_grad = True
        
        self.feature_extractor = self.base_model.features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension of MobileNetV3 large
        feature_dim = 960
        
        # Gender prediction branch
        self.gender_fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

        # BMI prediction branch
        self.bmi_fc = nn.Sequential(
            nn.Linear(feature_dim + 2, 512),  # Adding 2 features from gender prediction
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)   # Extract deep features
        features = self.global_pool(features)  # Pool to (batch_size, channels, 1, 1)
        features = torch.flatten(features, 1)  # Flatten to (batch_size, feature_dim)

        # Predict gender
        gender_logits = self.gender_fc(features)  
        predicted_gender = torch.softmax(gender_logits, dim=1)  # Convert to probability

        # Concatenate gender prediction with features for BMI prediction
        combined_input = torch.cat((features, predicted_gender), dim=1)

        bmi = self.bmi_fc(combined_input)  # Predict BMI

        return predicted_gender, bmi
    
    @property
    def is_cuda(self):
        """ Check if model parameters are allocated on the GPU. """
        return next(self.parameters()).is_cuda
    
    def save(self, path):
        """ Save model state dictionary. """
        print(f'Saving model state to {path}')
        torch.save(self.state_dict(), path)
        
    def export(self, model, device, file_name="bmi_model.onnx"):
        """ Export model to onnx format. """
        example_input = torch.randn(1, 3, 224, 224).to(device)
        torch.onnx.export(model, example_input, file_name, opset_version=11, export_params=True)
