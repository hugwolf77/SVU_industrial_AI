import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import torchsummary

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()else "cpu")

print(f"Using {device} device")

trainset = torchvision.datasets.MNIST(root = "", train = True, download = True, transform=torchvision.transforms.ToTensor()) 
testset = torchvision.datasets.MNIST(root = "", train = False, download = True, transform=torchvision.transforms.ToTensor()) 
print('number of training data : ', len(trainset)) 
print('number of test data : ', len(testset))

# train_loader = torch.utils.data.DataLoader(trainset,
#                                           batch_size=32,
#                                           shuffle=True,
#                                           num_workers=1)
# test_loader = torch.utils.data.DataLoader(testset,
#                                           batch_size=32,
#                                             shuffle=True,
#                                           `num_workers=1`)

train_loader = DataLoader(trainset,
                            batch_size=32,
                            shuffle=True)
test_loader = DataLoader(testset,
                            batch_size=32,
                            shuffle=True)

class_names = ['0','1','2','3','4','5','6','7','8','9'] 
plt.figure(figsize=(10, 10)) 
images, labels = next(iter(train_loader)) 
for i in range(16): 
	ax = plt.subplot(4, 4, i + 1) 
	img = images[i] 
	img = (img - img.min())/(img.max() - img.min()) 
	plt.imshow(torch.squeeze(img.permute(1, 2, 0)).numpy()) 
	plt.title(f'Class:{class_names[labels[i]]}') 
	plt.axis("off")
plt.show()

class MLP(nn.Module): 
	def __init__(self): 
		super(MLP,self).__init__() 
		self.linear_1 = nn.Linear(784,512) 
		self.relu_1 = nn.ReLU()
		self.linear_2 = nn.Linear(512,256) 
		self.relu_2 = nn.ReLU() 
		self.final = nn.Linear(256, 10) 
		
	def forward(self, x): 
		x = x.view(-1, 28 * 28) 
		x = self.linear_1(x) 
		x = self.relu_1(x)
		x = self.linear_2(x) 
		x = self.relu_2(x) 
		x = self.final(x) 
		return x

class LRScheduler():
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor # LR을 factor배로 감소시킴
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class EarlyStopping():
    def __init__(self, patience=5, verbose=False, delta=0, path=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False # 조기 종료를 의미하며 초기값은 False로 설정
        self.delta = delta # 오차가 개선되고 있다고 판단하기 위한 최소 변화량
        self.path = path
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        # 에포크 만큼 한습이 반복되면서 best_loss가 갱신되고, bset_loss에 진전이 없으면 조기종료 후 모델을 저장
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Early Stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}. Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"set device = {device}")
model = MLP().to(device) 
criterion = torch.nn.CrossEntropyLoss().to(device) 
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1)

save_path = "./exp_save_01.pth"
lr_scheduler = LRScheduler(optimizer=optimizer, patience=5, min_lr=1e-4, factor=0.5)
early_stopping = EarlyStopping(patience=5, verbose=False, delta=0, path=save_path)



torchsummary.summary(model,(1, 28, 28))


# train
epochs = 100
torch.cuda.empty_cache()   
train_losses = []
test_losses = []
train_acc = []
test_acc = []

for e in range(epochs):
    # training loop
    running_loss = 0       
    running_accuracy = 0 
    model.train()
    for _, data in enumerate(tqdm(train_loader)):
        # training phase            
        inputs, labels = data
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        optimizer.zero_grad()  # reset gradient
        
        # forward        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)            

        # backward
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_accuracy += torch.sum(preds == labels.data).detach().cpu().numpy()/inputs.size(0)


    model.eval()
    val_loss = 0
    val_accuracy = 0
    # validation loop
    with torch.no_grad():
        for _, data in enumerate(tqdm(test_loader)):                
            inputs, labels = data
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # evaluation metrics
            # loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy += torch.sum(preds == labels.data).detach().cpu().numpy()/inputs.size(0)
            lr_scheduler(val_loss)
            early_stopping(val_loss, model)

    # calculate mean for each batch
    train_losses.append(running_loss / len(train_loader))
    test_losses.append(val_loss / len(test_loader))
    train_acc.append(running_accuracy / len(train_loader))
    test_acc.append(val_accuracy / len(test_loader))
    print("Epoch:{}/{}..".format(e + 1, epochs),
            "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
            "Val Loss: {:.3f}..".format(val_loss / len(test_loader)),
            "Train Acc:{:.3f}..".format(running_accuracy / len(train_loader)),
            "Val Acc:{:.3f}..".format(val_accuracy / len(test_loader)))

history = {'train_loss': train_losses, 'val_loss': test_losses,
            'train_acc': train_acc, 'val_acc': test_acc}
