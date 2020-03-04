import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import os.path
import csvmaker as csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as transform
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import output.svr as reg
from datetime import datetime
from datetime import timedelta

num_epochs = 100
train_batchsize = 10
test_batchsize = 8
cl_num = 20

# CUDA?
cuda = torch.cuda.is_available()

if cuda:
    print("Training with GPU....")

class CustomDataset(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data_info = pd.read_csv(csv_path, header=None)    # Read the csv file
        self.image_arr = np.asarray(self.data_info.iloc[:, 0]) # 1st column contains the csv paths with image names
        self.clfeatures_arr = np.asarray(self.data_info.iloc[:, 1:cl_num*6+1]).astype(np.float32) # 2nd~7th columns contain features of colorline 
        self.label_arr = np.asarray(self.data_info.iloc[:, cl_num*6+1:cl_num*6+4]).astype(np.float32) #  8th~10th column are the labels (R,G,B values)
        self.data_len = len(self.data_info.index) # Calculate len

    def __getitem__(self, index):
       
        single_image_name = self.image_arr[index] # Get image name from the pandas df
        cl_features = self.clfeatures_arr[index]   # Get the features of colorline
        single_image_label = self.label_arr[index]  # Get RGB labels of the images 
        return (cl_features, single_image_label, single_image_name)  # Return images and labels

    def __len__(self):
        return self.data_len

#MLP module for color estimation of illumination   
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(cl_num*6,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,3)
        
    def forward(self,din):
        din = din.view(-1, 6)
        dout = nn.functional.relu(self.fc1(din))
        dout = nn.functional.relu(self.fc2(dout))
        dout = nn.functional.relu(self.fc3(dout))
        return self.fc4(dout)        

if __name__ == "__main__":
    # train dataset
    train_data = CustomDataset('dataset/train_input.csv') 
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=train_batchsize, shuffle=True, drop_last=True)
    test_data = CustomDataset('dataset/test_input.csv') 
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=test_batchsize, shuffle=False)

    model = MLP()
    if cuda:
        model.cuda()
        
    criterion = nn.MSELoss()
    if cuda:
        criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    
    best_accuracy = torch.FloatTensor([180])
    best_accuracy_test = torch.FloatTensor([180])
    best_trainregAcc, best_regAcc  = 180, 180
    start_epoch = 0
    train_acc_list, test_acc_list = [], []
    trainregAcclist,regAcclist = [], []
    checkpoint_time = timedelta(0,0)
    resume_weights = "checkpoint/checkpoint.pth.tar" # Path to saved model weights

    def save_checkpoint(state, is_best, filename='checkpoint/checkpoint.pth.tar'):
        """Save checkpoint if a new best is achieved"""
        if is_best:
            print("=> Saving a new best\n")
            torch.save(state, filename)  # save checkpoint
            
        else:
            print("=> Validation Accuracy did not improve\n")

    # If exists a best model, load its weights!
    if os.path.isfile(resume_weights):
        print("=> loading checkpoint '{}' ...".format(resume_weights))
        if cuda:
            checkpoint = torch.load(resume_weights)
        else:
            checkpoint = torch.load(resume_weights, map_location=lambda storage,loc: storage)
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        best_accuracy_test = checkpoint['best_accuracy_test']
        best_regAcc = checkpoint['best_regAcc']
        best_trainregAcc = checkpoint['best_trainregAcc'] 
        train_acc_list = checkpoint['train_acc_list']
        test_acc_list = checkpoint['test_acc_list']
        checkpoint_time = checkpoint['time']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights,checkpoint['epoch']))

    total_step = len(train_loader) #size of train data
    total_step_test = len(test_loader) #size of test data

    #Angular error calculation
    def error(output,label):  
        upper = (np.dot(output, label))
        lower = math.sqrt(math.fsum(output**2))*math.sqrt(math.fsum(label**2))
        rad = np.clip((upper/lower),-1.0,1.0)
        acc = math.degrees(math.acos(rad))
        return acc

    def train(epoch):
        print('Start Training\n')
        model.train()
        correct, total = 0.0, 0
        tmp_output, tmp_path, tmp_error, = [], [], []

        for i, (images, labels, path) in enumerate(train_loader, 0):
            # get the inputs and labels
            inputs = Variable(images)
            labels = Variable(labels)
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()    # zero the parameter gradients
            # forward + backward + optimize
            outputs = model(inputs) #outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            if cuda: # Load loss on CPU
                loss.cpu()
            loss.backward()
            optimizer.step()

            a = np.squeeze(labels.cpu().numpy())
            b = np.squeeze(outputs.cpu().detach().numpy())
            tmp_output.append(b)
            tmp_path.append(path)
            
            # Track the accuracy
            if train_batchsize==1:
                total += labels.size(0)
                correct += error(a,b)
                tmp_error.append(error(a,b))
            else:
                total += labels.size(0)
                for j in range(labels.size(0)):
                    correct += error(a[j],b[j])
                    tmp_error.append(error(a[j],b[j]))

            if (i + 1) % 1000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}, Angular error: {:.6f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),(correct/total)))
        
        print('Average Angular error: {:.6f}'.format(correct / total))
        print('Finished Training')
        train_acc_list.append(correct/total)
        acc = torch.FloatTensor([correct/total])
        return acc, tmp_output, tmp_path, tmp_error
    
    def test():
        print('Start Testing\n')
        model.eval()
        correct, total = 0.0, 0
        tmp_output, tmp_path, tmp_error = [], [], []

        for i, (images, labels, path) in enumerate(test_loader, 0):
            # get the inputs and labels
            inputs = Variable(images)
            labels = Variable(labels)
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            if cuda: # Load loss on CPU
                loss.cpu()
            a = np.squeeze(labels.cpu().numpy())
            b = np.squeeze(outputs.cpu().detach().numpy())
            tmp_output.append(b)
            tmp_path.append(path)

            # Track the accuracy
            if test_batchsize==1:
                total += labels.size(0)
                correct += error(a,b)
                tmp_error.append(error(a,b))
            else:
                total += labels.size(0)
                for j in range(labels.size(0)):
                    correct += error(a[j],b[j])
                    tmp_error.append(error(a[j],b[j]))

            if (i + 1) % 1000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}, Angular error: {:.6f}'.format(epoch + 1, num_epochs, i + 1, total_step_test, loss.item(), (correct/total)))
        
        print('Average Angular error: {:.6f}'.format(correct / total))
        print('Finished Testing\n')
        acc = torch.FloatTensor([correct/total]) 
        test_acc_list.append(correct/total)
        return acc, tmp_output, tmp_path, tmp_error 

    start = datetime.now() # Calculation of time used 
 
    for epoch in range(start_epoch, num_epochs):
        train_output, test_output = [], []
        train_path, test_path = [], []
        train_error, test_error = [], []

        acc, train_output, train_path, train_error = train(epoch)
        acc1, test_output, test_path, test_error  = test()
        csv.outputs_file(train_output, test_output, train_path, test_path, train_batchsize, test_batchsize, train_error, test_error, epoch+1, "")
        trainregAcc, regAcc = reg.svr(best_regAcc)
        trainregAcclist.append(trainregAcc)
        regAcclist.append(regAcc)
        
        is_best = bool(regAcc < best_regAcc)  # Get bool
        if is_best == True: 
            csv.outputs_file(train_output, test_output, train_path, test_path, train_batchsize, test_batchsize, train_error, test_error, epoch+1, "regbest_")

        if bool(acc1 < best_accuracy_test) == True: 
            csv.outputs_file(train_output, test_output, train_path, test_path, train_batchsize, test_batchsize, train_error, test_error, epoch+1, "perpatch_best_")

        best_accuracy = torch.FloatTensor(min(acc.numpy(), best_accuracy.numpy())) # keep track best acc
        best_accuracy_test = torch.FloatTensor(min(acc1.numpy(), best_accuracy_test.numpy())) #  keep track best acc
        best_regAcc = min(regAcc, best_regAcc) #keep track best acc
        best_trainregAcc = min(trainregAcc, best_trainregAcc) #keep track best acc
	    
        # Save checkpoint if is a new best
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_accuracy': best_accuracy, 'best_accuracy_test': best_accuracy_test,
                        'best_regAcc': best_regAcc,'best_trainregAcc': best_trainregAcc,'test_acc_list': test_acc_list, 'train_acc_list': train_acc_list, 
                        'time': datetime.now()-start}, is_best)

    print('Time used: {}'.format(datetime.now() - start + checkpoint_time))
    print('Final Average Angular Error (train): {:.6f}'.format(np.mean(trainregAcclist)))
    print('Final Average Angular Error (test): {:.6f}'.format(np.mean(regAcclist)))
    print('Best Angular Error (train): {}'.format(best_trainregAcc))
    print('Best Angular Error (test): {}'.format(best_regAcc))

    epochaxis = [t for t in range(num_epochs)]
    # plot with various axes scales
    plt.figure(1)
    plt.subplot(221)
    plt.plot(epochaxis, train_acc_list, 'r--', epochaxis, test_acc_list, 'b--')
    plt.xlabel('Epoch')
    plt.ylabel('Angular error')
    plt.title('Average Angular Error without Regression ')
    plt.grid(True)

    plt.subplot(222)
    plt.plot(epochaxis, trainregAcclist, 'r--', epochaxis, regAcclist, 'b--')
    plt.xlabel('Epoch')
    plt.ylabel('Angular error')
    plt.title('Average Angular Error with Regression')
    plt.grid(True)

    plt.show()