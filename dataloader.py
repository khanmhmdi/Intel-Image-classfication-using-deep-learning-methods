import numpy as np 
import torch
import torchvision

from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

class dataloader():
    def __init__(self , train_path , test_path , use_transform=False , batch_size=64 , valid_size=0.15) -> None:
        self.train_path = train_path 
        self.test_path = test_path
        self.use_transform = use_transform
        self.bathc_size = batch_size
        self.valid_size = valid_size

    def transform(self):
        train_transforms = torchvision.transforms.Compose([
        transforms.Resize((150,150)),
        transforms.RandomHorizontalFlip(p=0.5), # randomly flip and rotate
        transforms.ColorJitter(0.3,0.4,0.4,0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
        ])

        # Augmentation on test images not needed
        transform_tests = torchvision.transforms.Compose([
            transforms.Resize((150,150)),
            transforms.ToTensor(),
            transforms.Normalize((0.425, 0.415, 0.405), (0.255, 0.245, 0.235))
            ])

        return train_transforms , transform_tests

    def image_folder_loader(self , train_transforms , transform_tests):

        train_dataset = datasets.ImageFolder(self.train_path, transform = train_transforms)
        test_dataset = datasets.ImageFolder(self.test_path, transform = transform_tests)

        return train_dataset , test_dataset 

    def split_train_valid(self , train_dataset ,test_dataset):

        num_train = len(train_dataset) #get the length of the train dataset
        indicies = list(range(num_train)) # list [0,...,length of the train dataset]
        np.random.shuffle(indicies) # shuffle the above list
        split = int(self.valid_size * num_train) # what is split number for the train and valid index
        train_idx, valid_idx = indicies[split : ], indicies[ : split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        return train_sampler , valid_sampler

    def dataloader_pytorch(self, train_dataset , test_dataset , train_sampler , valid_sampler):

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size
                                                ,num_workers = 2, sampler = train_sampler) 
        valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size
                                                ,num_workers = 2, sampler = valid_sampler) 
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = self.batch_size
                                                ,num_workers = 2, drop_last = True) 

        return train_loader , valid_loader , test_loader

    def load_data(self):
        
        train_transforms ,transform_tests =  self.transforms()
        train_dataset , test_dataset  = self.image_folder_loader(train_transforms=train_transforms , transform_tests=transform_tests)
        train_sampler , valid_sampler = self.split_train_valid(train_dataset=train_dataset , test_dataset=test_dataset)
        train_loader , valid_loader , test_loader = self.dataloader_pytorch(train_dataset=train_dataset , test_dataset=test_dataset , train_sampler=train_sampler , valid_sampler=valid_sampler)
        
        return train_loader , valid_loader , test_loader