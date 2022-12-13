import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class trainer:
    def __init__(self, model, epoch, learning_rate, optimizer, loss):
        """Initializing parameters."""
        self.model = model
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criterion = loss
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_total_step = None
        self.val_total_step = None

        self.val_loss = []
        self.val_acc = []
        self.train_loss = []
        self.train_acc = []

    def train(self , train_dataset ,valid_dataset):
        for epoch in range(self.epoch):
            running_loss = 0.0
            correct = 0

            loss_val = 0.0
            correct_val = 0

            self.model.train()
            avg_loss ,avg_acc = self.run_train_epoch(train_dataset=train_dataset,running_loss=running_loss,correct=correct)
            self.model.eval()
            avg_loss_val ,avg_acc_val = self.run_eval_epoch(valid_dataset=valid_dataset, loss_val=loss_val , correct_val=correct_val)

            print('[epoch %d] loss: %.5f accuracy: %.4f val loss: %.5f val accuracy: %.4f' % (epoch + 1, avg_loss, avg_acc, avg_loss_val, avg_acc_val))

    def run_train_epoch(self,train_dataset,running_loss,correct):
        
        for data in train_dataset:
            batch, labels = data
            batch = batch.float()
            batch, labels = batch.to(self.device), labels.to(self.device)
            # batch = weights.transforms(bacth)

            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # compute training statistics
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        avg_loss = running_loss /len(train_dataset)
        avg_acc = correct /len(train_dataset)

        self.train_loss.append(avg_loss)
        self.train_acc.append(avg_acc)

        return avg_loss ,avg_acc  

    def run_eval_epoch(self, valid_dataset,loss_val , correct_val):
        with torch.no_grad():
            
            for data in valid_dataset:
                batch, labels = data
                batch = batch.float()
                batch, labels = batch.to(self.device), labels.to(self.device)

                outputs = self.model(batch)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                
                correct_val += (predicted == labels).sum().item()
                loss_val += loss.item()
                
                avg_loss_val = loss_val / int(len(valid_dataset))
                avg_acc_val = correct_val / int(len(valid_dataset))
                
                self.val_loss.append(avg_loss_val)
                self.val_acc.append(avg_acc_val)

        return avg_loss_val ,avg_acc_val        