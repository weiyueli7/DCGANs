import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms  
import torchvision
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import cv2
import pandas as pd
import torchvision.transforms as transforms 
from torchvision.transforms import ToTensor,Normalize, RandomHorizontalFlip, Resize
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

from resnet import*


parser = argparse.ArgumentParser(description = 'Train ResNet')
parser.add_argument("--mode", type=int, default=0, help="0: fake only; 1: real only; 2: both")
parser.add_argument("--num_epochs", type=int, default=7, help="number of epochs of training")
parser.add_argument("--train_batch_size", type=int, default=100, help="size of the train batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
hyperparameters = parser.parse_args()
hyperparameters = list(vars(hyperparameters).values())
hyperparameters = tuple(hyperparameters)



mode = hyperparameters[0]


if not os.path.exists('graphs/'):
    os.makedirs('graphs')

data_dir_Train = "dataset/seg_train_real"
data_dir_Train_fake ="fake"
data_dir_Test = "data/seg_test"
data_dir_pred = "data/seg_pred/seg_pred"

train_dir = data_dir_Train + "/seg_train/"#""
valid_dir = data_dir_Test + "/seg_test/"
pred_files = [os.path.join(data_dir_pred, f) for f in os.listdir(data_dir_pred)]

outcomes = os.listdir(train_dir)




# convert data to a normalized torch.FloatTensor
transform = torchvision.transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.RandomHorizontalFlip(p=0.5), # randomly flip and rotate
    transforms.ColorJitter(0.3,0.4,0.4,0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
    ])

# Augmentation on test images not needed
transform_tests = torchvision.transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.425, 0.415, 0.405), (0.255, 0.245, 0.235))
    ])

if mode == 0:
    des = "all fake"
    train_data = torchvision.datasets.ImageFolder(root=data_dir_Train_fake,transform=transform)
elif mode == 1:
    des = "all real"
    train_data = torchvision.datasets.ImageFolder(root=train_dir,transform=transform)
else:
    des = "all fake + real"
    l=[]
    l.append(torchvision.datasets.ImageFolder(data_dir_Train_fake, transform=transform))
    l.append(torchvision.datasets.ImageFolder(train_dir, transform=transform))
    train_data = torch.utils.data.ConcatDataset(l)

test_data = torchvision.datasets.ImageFolder(root=valid_dir,transform=transform_tests)


valid_size = 0.15
# Splot data into train and validation set
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


train_loader = DataLoader(train_data,batch_size=hyperparameters[2],sampler=train_sampler,num_workers=2)
valid_loader = DataLoader(train_data, batch_size =100, sampler=valid_sampler, num_workers=3)
test_loader= DataLoader(test_data,batch_size=32,shuffle=False,num_workers=2)


train_on_gpu = torch.cuda.is_available()
device =  torch.device('cuda' if torch.cuda.is_available else 'cpu')

model = resnet(len(outcomes))
model.to(device)


criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters[3])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,6], gamma=0.06)


epochs = hyperparameters[1]

# track change in validation loss
valid_loss_min = np.Inf
val_loss = []
tn_loss = []
for epoch in range(1,epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    # Train the model
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):       
        # move tensor to gpu if cuda is available
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        # clear the gradiant of all optimizer variable
        optimizer.zero_grad()
        # forward pass: compute pradictions by passing inputs
        output = model(data)
        # calculate batch loss
        loss = criterion(output, target)
        # backward pass: compute gradiant of the loss with respect to the parameters
        loss.backward()
        # update parameters by optimizing single step
        optimizer.step()
        
        # update training loss
        train_loss += loss.item()*data.size(0)

    # validate the model

    model.eval()
    for batch_idx, (data, target) in enumerate(valid_loader):
        # move tensor to gpu
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        # forward pass: compute the validation predictions
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update the validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average loss
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    val_loss.append(valid_loss)
    tn_loss.append(train_loss)
    # update learning rate
    scheduler.step()
    # Print the train and validation loss statistic
    print('Epoch: {} \t Training Loss: {:.3f} \t Validation Loss: {:.3f}'.format(epoch, train_loss, valid_loss))
    
    # save model if validation loss decrease
    if valid_loss <= valid_loss_min:
        print("Validation loss decreased {:.4f}--->{:.4f}  Saving model...".format(valid_loss_min, valid_loss))
        # save current model
        torch.save(model.state_dict(), 'model_state.pt')
        valid_loss_min = valid_loss
    print('Learning Rate ------------->{:.4f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
    

plt.plot(tn_loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.legend(frameon=False)
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.title('ResNet Training and Validation Loss: ' + des)
plt.savefig(f"graphs/{des}_loss.png")

model.load_state_dict(torch.load('model_state.pt'))
model.eval()
model.cuda()

correct_count, all_count = 0,0
for images, labels in test_loader:
    for i in range(len(labels)):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        img = images[i].view(1,3,64,64)
        with torch.no_grad():
            logps = model(img)
            
        ps = torch.exp(logps)
        probab = list(ps.cpu()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.cpu()[i]
        if(true_label == pred_label):
            correct_count += 1
        all_count += 1
        
print("Number of images Tested=", all_count)
print("\n Model Accuracy=",(correct_count/all_count)*100)

def weight_map(model):
    for i, weight in enumerate(model.parameters()):
        first_weight = weight
        break
    first_weight = first_weight.cpu().detach().numpy()
    # start plotting the weight map
    fig = plt.figure(figsize=(10, 10))
    for i, sub_weight in enumerate(first_weight):
        fig.add_subplot(8, 8, i + 1)
        # combine RGB
        sub_weight = np.dstack((sub_weight[0],sub_weight[1],sub_weight[2]))
        # normalize the weight
        sub_weight = sub_weight / 2 + 0.5
        sub_weight = sub_weight.clip(0,1)
        plt.imshow(sub_weight)
        plt.axis('off')
    fig.tight_layout(pad=1)
    fig.show()
    # save the weight plot
    fig.savefig('graphs/resnet_weights.png', bbox_inches = 'tight')
    
def feature_map(model, img):
    """
    Helper method to generate feature maps for `custom`, `resnet`, and `vgg` models
    """
    # same transform in `data.py`
    transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ]
    )
    # get the first image in the testing set
    image = img#get_dataset(img, transform)[0][0].reshape(1,3,224,224).cuda()
    # get all layers if the model is `resnet`
    if isinstance(model, resnet):
        conv_layer = []
        conv_layer += list(list(model.children())[0].children())[0:4]
        conv_layer += [i for i in list(list(list(list(model.children())[0].children())[4].children())[0].children())]
        conv_layer += [i for i in list(list(list(list(model.children())[0].children())[4].children())[1].children())]
        conv_layer += [i for i in list(list(list(list(model.children())[0].children())[5].children())[0].children())[0:-1]]      
        conv_layer += [i for i in list(list(list(list(model.children())[0].children())[5].children())[1].children())]
        conv_layer += [i for i in list(list(list(list(model.children())[0].children())[6].children())[0].children())[0:-1]]   
        conv_layer += [i for i in list(list(list(list(model.children())[0].children())[6].children())[1].children())]
        conv_layer += [i for i in list(list(list(list(model.children())[0].children())[7].children())[0].children())[0:-1]]
        conv_layer += [i for i in list(list(list(list(model.children())[0].children())[7].children())[1].children())]
        conv_layer += list(list(model.children())[0].children())[8:]

    # select the initial, middle, and last layer if model is `resnet`
    if isinstance(model, resnet):
        outputs = []
        for i, layer in enumerate(conv_layer):
            if i == 45:
                image = layer(image.view(-1, 512))
            else:
                image = layer(image)
            if i == 0 or i == 22 or i == 42:
                outputs.append(image)
    # draw three plots 
    for i in range(3):
        
        processed = []
        
        for feature_map in outputs[i][0]:
            feature_map = feature_map / feature_map.shape[0]
            processed.append(feature_map.data.cpu().numpy())
        # initialize plot
        fig = plt.figure(figsize=(40, 40))
        # plot a resnet model's feature maps
        if isinstance(model, resnet):
            # plot the first layer
            if i == 0:
                for idx in range(len(processed)):
                    a = fig.add_subplot(8, 8, idx+1)
                    imgplot = plt.imshow(processed[idx])
                    a.axis("off")
                plt.savefig("graphs/" + "resnet_initial_layer_feature.png")
            # plot the middle layer
            elif i == 1:
                for idx in range(len(processed)):
                    a = fig.add_subplot(16, 16, idx+1)
                    imgplot = plt.imshow(processed[idx])
                    a.axis("off")
                plt.savefig("graphs/" + "resnet_middle_layer_feature.png")
            # plot the last layer
            else:
                for idx in range(len(processed)):
                    a = fig.add_subplot(32, 16, idx+1)
                    imgplot = plt.imshow(processed[idx])
                    a.axis("off")
                plt.savefig("graphs/" + "resnet_final_layer_feature.png")
        # do not plot if the model is baseline
        else:
            return "No Model Found."

weight_map(model)
transform = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ]
)
# Open the image form working directory
target_image = Image.open('dataset/seg_test/seg_test/mountain/20093.jpg')
target_image = transform(target_image).reshape(1,3,64,64).cuda()
feature_map(model, target_image)