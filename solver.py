import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

    
class Solver(object):
    def __init__(self):
        self.optim_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.001}
        self.optim = torch.optim.AdamW
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.criterion_gender = nn.CrossEntropyLoss()  # For gender classification
        self.criterion_bmi = nn.SmoothL1Loss()         # For BMI regression
    
    def train(self, model, train_dataloader, val_dataloader, num_epochs=50):
      
        optim = self.optim(model.parameters(), **self.optim_args)
        scheduler = StepLR(optim, step_size=25, gamma=0.1)
        model.to(self.device)

        print('START TRAIN.')
        epoch_loss_train = []
        epoch_loss_val = []
        
        for epoch in range(num_epochs):
            # TRAINING
            total_loss_train = 0.0
            model.train()
            for images, genders, targets in train_dataloader:
                images, genders, targets = images.to(self.device), genders.to(self.device), targets.to(self.device)
                
                optim.zero_grad()
                predicted_genders, predicted_bmis = model(images)

                # Compute loss
                loss_gender = self.criterion_gender(predicted_genders, genders)  # Classification loss
                loss_bmi = self.criterion_bmi(predicted_bmis.squeeze(), targets)  # Regression loss
                
                total_loss = loss_gender + 5*loss_bmi  # weighted sum
                
                total_loss.backward()
                optim.step()

               
                total_loss_train += total_loss.item()*images.size(0)
             
            scheduler.step()
    
            epoch_loss_train.append(total_loss_train/len(train_dataloader.sampler))
            
            total_loss_val = 0.0
            model.eval()
            with torch.no_grad():
                for images, genders, targets in val_dataloader:
                    images, genders, targets = images.to(self.device), genders.to(self.device), targets.to(self.device)
                    predicted_genders, predicted_bmis = model(images)

                    loss_gender = self.criterion_gender(predicted_genders, genders)  # Classification loss
                    loss_bmi = self.criterion_bmi(predicted_bmis.squeeze(), targets)  # Regression loss

                    total_loss = loss_gender + 5*loss_bmi
                
                    total_loss_val += total_loss.item()*images.size(0)

            epoch_loss_val.append(total_loss_val/len(val_dataloader.sampler))
            
            print('Epoch ',epoch, ' Train Loss ',total_loss_train/len(train_dataloader.sampler), ' Val Loss ',total_loss_val/len(val_dataloader.sampler), ' LR ',optim.param_groups[0]['lr']) 
       
        plt.figure(0)
        plt.plot(np.array(epoch_loss_train), label ='Train loss')
        plt.plot(np.array(epoch_loss_val), label ='Val loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train and Validation loss curves')
        plt.grid(True)

        print('FINISH.')