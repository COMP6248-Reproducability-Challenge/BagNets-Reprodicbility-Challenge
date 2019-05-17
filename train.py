import torch
from torchvision import transforms
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from BagNet_model import bagnet33

# init the BagNet instance
MyBagNetModel = bagnet33(num_classes=120)

# assign adaptive device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MyBagNetModel.to(DEVICE)


torch.cuda.manual_seed(7)
IMAGE_PATH = '/content/drive/My Drive/DL_Coursework/Images'
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.299, 0.224, 0.225]),
                                ])
# create the dataset we need, here is Stanford Dogs from kaggle
Stanford_Dogs_dataset = ImageFolder(f'{IMAGE_PATH}', transform=TRANSFORM)


train_size = 8200
valid_size = 1600
test_size = len(Stanford_Dogs_dataset) - train_size - valid_size

train_dataset, test_dataset = random_split(Stanford_Dogs_dataset, lengths=(train_size+valid_size,test_size))
train_dataset, valid_dataset = random_split(train_dataset,(train_size, valid_size))

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset,batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False)

EPOCH_NUM = 50
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(MyBagNetModel.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                       amsgrad=False)

if __name__ == '__main__':

    for epoch in range(EPOCH_NUM):
        tqdm_train = tqdm(train_loader)
        loss_train = []
        loss_valid = []
        MyBagNetModel.train()
        for train_data, label in tqdm_train:
            train_data = train_data.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            output = MyBagNetModel(train_data)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            loss_train.append(loss.detach().cpu().item())
            tqdm_train.set_postfix(loss=np.mean(loss_train), epoch=epoch)
            