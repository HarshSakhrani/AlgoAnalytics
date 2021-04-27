from pycm import *
import random
from audiomentations import Compose,TimeStretch,Shift
import sys
from statistics import mean 
from glob import glob  
from pydub import AudioSegment
import fnmatch
import math
import shutil
import os
import pandas as pd
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
import torch
import torchvision 
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataset
import torch.utils.data.dataloader
import torchaudio.transforms as audioTransforms
import torchvision.transforms as visionTransforms
import PIL.Image as Image
from torchvision.transforms import ToTensor,ToPILImage

pd.set_option('display.max_colwidth', -1)

dfClassification=pd.read_csv("80Classes/VastAIClassification_80Samples.csv",index_col=0)
dfTriplet=pd.read_csv("80Classes/VastAiTriplets_80classes.csv",index_col=0)
dfClassification=dfClassification.sample(frac=1).reset_index(drop=True) 
dfTriplet=dfTriplet.sample(frac=1).reset_index(drop=True)

dfClassificationTrain,dfClassificationVal,dfClassificationTest=np.split(dfClassification.sample(frac=1, random_state=42), [int(.8 * len(dfClassification)), int(.9 * len(dfClassification))])

dfClassificationTrain=dfClassificationTrain.reset_index(drop=True)
dfClassificationTest=dfClassificationTest.reset_index(drop=True)
dfClassificationVal=dfClassificationVal.reset_index(drop=True)

dfTripletTrain,dfTripletVal,dfTripletTest=np.split(dfTriplet.sample(frac=1, random_state=42), [int(.8 * len(dfTriplet)), int(.9 * len(dfTriplet))])

dfTripletTrain=dfTripletTrain.reset_index(drop=True)
dfTripletTest=dfTripletTest.reset_index(drop=True)
dfTripletVal=dfTripletVal.reset_index(drop=True)

from torch.utils.data import WeightedRandomSampler
freqLabels=torch.tensor(dfClassificationTrain['Label'].value_counts().sort_index(),dtype=torch.double)
weightClass=freqLabels/freqLabels.sum()
weightClass= 1/weightClass
weightClass=(weightClass).tolist()
sampleWeights=[weightClass[i] for i in dfClassificationTrain['Label']]
trainSampler=WeightedRandomSampler(sampleWeights,len(dfClassificationTrain))

#Without DALI. For PreTrained VGG

from torch.utils.data import Dataset, DataLoader

class SpectogramImgClassificationTrainDataset(Dataset):

  def __init__(self,dataframe,vision_transform=None):
    self.data=dataframe
    self.vision_transform=vision_transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):

    temp=random.randint(0,1)

    augment = Compose([
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5,rollover=False)
    ])

    self.wavPath=str(self.data.iloc[idx,0])
    self.label=self.data.iloc[idx,1]

    self.signal,self.sr=torchaudio.load(self.wavPath)

    if (temp==1):
      self.signal=torch.from_numpy(augment(samples=self.signal.numpy(),sample_rate=self.sr))
    
    self.spectogram=torchaudio.transforms.Spectrogram(n_fft=320,win_length=320,hop_length=160)(self.signal)
    self.logSpectogram=torchaudio.transforms.AmplitudeToDB()(self.spectogram)

    #self.tempImg=torchvision.transforms.ToPILImage()(self.logSpectogram)
    #self.tempImg=self.tempImg.convert("RGB")
    #self.spectogramImageTensor=self.vision_transform(self.tempImg)

    return self.logSpectogram,self.label

#Without DALI. For PreTrained VGG

from torch.utils.data import Dataset, DataLoader

class SpectogramImgClassificationTestDataset(Dataset):

  def __init__(self,dataframe,vision_transform=None):
    self.data=dataframe
    self.vision_transform=vision_transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    self.wavPath=str(self.data.iloc[idx,0])
    self.label=self.data.iloc[idx,1]

    self.signal,self.sr=torchaudio.load(self.wavPath)

    self.spectogram=torchaudio.transforms.Spectrogram(n_fft=320,win_length=320,hop_length=160)(self.signal)
    self.logSpectogram=torchaudio.transforms.AmplitudeToDB()(self.spectogram)

    #self.tempImg=torchvision.transforms.ToPILImage()(self.logSpectogram)
    #self.tempImg=self.tempImg.convert("RGB")
    #self.spectogramImageTensor=self.vision_transform(self.tempImg)

    return self.logSpectogram,self.label

classificationTrainDataset=SpectogramImgClassificationTrainDataset(dataframe=dfClassificationTrain)
classificationTestDataset=SpectogramImgClassificationTestDataset(dataframe=dfClassificationTest)
classificationValDataset=SpectogramImgClassificationTestDataset(dataframe=dfClassificationVal)

classificationTrainLoader=torch.utils.data.DataLoader(classificationTrainDataset,batch_size=8,sampler=trainSampler)
classificationTestLoader=torch.utils.data.DataLoader(classificationTestDataset,batch_size=8,shuffle=True)
classificationValLoader=torch.utils.data.DataLoader(classificationValDataset,batch_size=8,shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomizedResNet(nn.Module):
  def __init__(self,originalResNet):
    super(CustomizedResNet,self).__init__()
    originalResNet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False)
    self.ResNetLayers=nn.Sequential(*(list(originalResNet.children())[:-2]))

    self.commonFC1=nn.Linear(107008,4096)
    #self.commonFC2=nn.Linear(16384,4096)
    self.softmaxFC1=nn.Linear(4096,80)
    self.tripletFC1=nn.Linear(4096,2048)
    self.tripletFC2=nn.Linear(2048,512)

  def forward(self,input,preTrainingFlag=0):
    convOutput=self.ResNetLayers(input)
    convOutput = convOutput.view(convOutput.size(0), -1)    

    if (preTrainingFlag==0):
      tripletOutput=self.commonFC1(convOutput)
      tripletOutput=nn.ReLU()(tripletOutput)
      tripletOutput=nn.Dropout()(tripletOutput)

      #tripletOutput=self.commonFC2(tripletOutput)
      #tripletOutput=nn.ReLU()(tripletOutput)
      #tripletOutput=nn.Dropout()(tripletOutput)

      tripletOutput=self.tripletFC1(tripletOutput)
      tripletOutput=nn.ReLU()(tripletOutput)
      tripletOutput=nn.Dropout()(tripletOutput)

      tripletOutput=self.tripletFC2(tripletOutput)
      #tripletOutput=nn.ReLU(inplace=True)(tripletOutput)
      #tripletOutput=nn.Dropout()(tripletOutput)

      return tripletOutput

    else:
      softmaxOutput=self.commonFC1(convOutput)
      softmaxOutput=nn.ReLU()(softmaxOutput)
      softmaxOutput=nn.Dropout()(softmaxOutput)

      #softmaxOutput=self.commonFC2(softmaxOutput)
      #softmaxOutput=nn.ReLU()(softmaxOutput)
      #softmaxOutput=nn.Dropout()(softmaxOutput)            

      softmaxOutput=self.softmaxFC1(softmaxOutput)
      #softmaxOutput=nn.ReLU(inplace=True)(softmaxOutput)
      #softmaxOutput=nn.Dropout()(softmaxOutput)      
    
      return softmaxOutput    

originalResNet=torchvision.models.resnet18()
model=CustomizedResNet(originalResNet)
model.to(device)
softmaxLoss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

def Average(lst): 
    return sum(lst) / len(lst) 

def train_model(model,patience,epochs):


  trainBatchCount=0
  testBatchCount=0

  avgTrainAcc=[]
  avgValidAcc=[]
  trainAcc=[]
  validAcc=[]
  trainLosses=[]
  validLosses=[]
  avgTrainLoss=[]
  avgValidLoss=[]

  #early_stopping = EarlyStopping(patience=patience, verbose=True)

  for i in range(epochs):

    print("Epoch:",i)

    model.train()
    print("Training.....")
    for batch_idx,(data,targets) in enumerate(classificationTrainLoader):

      trainBatchCount=trainBatchCount+1

      data=data.to(device)
      targets=targets.to(device)

      optimizer.zero_grad()

      scores=model(data,preTrainingFlag=1)
       
      loss=softmaxLoss(scores,targets)

      loss.backward()

      optimizer.step()

      trainLosses.append(float(loss))

      
      correct=0
      total=0
      total=len(targets)

      #print("Targets:-",targets)

      predictions=torch.argmax(scores,dim=1)

      #print("Predictions:-",predictions)

      correct = (predictions==targets).sum()
      acc=float((correct/float(total))*100)

      trainAcc.append(acc)

      if ((trainBatchCount%200)==0):

        print("Targets:-",targets)
        print("Predictions:-",predictions)

        print('Loss: {}  Accuracy: {} %'.format(loss.data, acc))

    model.eval()
    print("Validating.....")
    for data,targets in classificationValLoader:

      testBatchCount=testBatchCount+1

      data=data.to(device=device)
      targets=targets.to(device=device)

      scores=model(data,preTrainingFlag=1)

      loss=softmaxLoss(scores,targets)

      validLosses.append(float(loss))

      testCorrect=0
      testTotal=0

      _,predictions=scores.max(1)

      testCorrect = (predictions==targets).sum()
      testTotal=predictions.size(0)

      testAcc=float((testCorrect/float(testTotal))*100)

      validAcc.append(testAcc)

      if ((testBatchCount%200)==0):

        #print("Targets:-",targets)
        #print("Predictions:-",predictions)

        print('Loss: {}  Accuracy: {} %'.format(float(loss), testAcc))
    

    trainLoss=Average(trainLosses)
    validLoss=Average(validLosses)
    avgTrainLoss.append(trainLoss)
    avgValidLoss.append(validLoss)
    tempTrainAcc=Average(trainAcc)
    tempTestAcc=Average(validAcc)
    avgTrainAcc.append(tempTrainAcc)
    avgValidAcc.append(tempTestAcc)

    print("Epoch Number:-",i,"  ","Training Loss:-"," ",trainLoss,"Validation Loss:-"," ",validLoss,"Training Acc:-"," ",tempTrainAcc,"Validation Acc:-"," ",tempTestAcc)

    trainAcc=[]
    ValidAcc=[]
    trainLosses=[]
    validLosses=[]

    #early_stopping(validLoss,model)

    #if early_stopping.early_stop:
      #print("Early Stopping")
      #break

  #model.load_state_dict(torch.load("checkpoint.pt"))

  return model,avgTrainLoss,avgValidLoss,avgTrainAcc,avgValidAcc


model,avgTrainLoss,avgValidLoss,avgTrainAcc,avgValidAcc=train_model(model,3,10)

loss_train = avgTrainLoss
loss_val = avgValidLoss
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss(Pre-Training/ResNet-18)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('ResNet18_preTrainingLoss.png', bbox_inches='tight')
plt.close()
#plt.show()

loss_train = avgTrainAcc
loss_val = avgValidAcc
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training Acc')
plt.plot(epochs, loss_val, 'b', label='Validation Acc')
plt.title('Training and Validation Accuracy(Pre-Training/ResNet-18)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('ResNet18_preTrainingAcc.png', bbox_inches='tight')
plt.close()

def checkClassificationMetrics(loader,model):

  completeTargets=[]
  completePreds=[]

  correct=0
  total=0
  model.eval()

  with torch.no_grad():
    for data,targets in loader:
      data=data.to(device=device)
      targets=targets.to(device=device)

      scores=model(data,preTrainingFlag=1)
      _,predictions=scores.max(1)

      targets=targets.tolist()
      predictions=predictions.tolist()

      completeTargets.append(targets)
      completePreds.append(predictions)
      #correct += (predictions==targets).sum()
      #total+=predictions.size(0)


    completeTargetsFlattened=[item for sublist in completeTargets for item in sublist]
    completePredsFlattened=[item for sublist in completePreds for item in sublist]

    cm = ConfusionMatrix(actual_vector=completeTargetsFlattened, predict_vector=completePredsFlattened)
    return cm

softmaxCM=checkClassificationMetrics(classificationTestLoader,model)

f=open("softmaxResults.txt","a")
f.write(str(softmaxCM))
f.close()

print("=================================================================================================================================")
print("Fine-Tuning")
print("=================================================================================================================================")

class SpectogramImgTripletTrainDataset(Dataset):

  def __init__(self,dataframe,vision_transform=None):
    self.data=dataframe
    self.vision_transform=vision_transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):

    augment = Compose([
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5,rollover=False)
    ])

    temp1=random.randint(0,1)
    temp2=random.randint(0,1)
    temp3=random.randint(0,1)

    self.anchor=str(self.data.iloc[idx,0])
    self.positive=self.data.iloc[idx,1]
    self.negative=self.data.iloc[idx,2]

    self.signalAnchor,self.srAnchor=torchaudio.load(self.anchor)
    self.signalPositive,self.srPositive=torchaudio.load(self.positive)
    self.signalNegative,self.srNegative=torchaudio.load(self.negative)

    if (temp1==1):
      self.signalAnchor=torch.from_numpy(augment(samples=self.signalAnchor.numpy(),sample_rate=self.srAnchor))

    if (temp2==1):
      self.signalPositive=torch.from_numpy(augment(samples=self.signalPositive.numpy(),sample_rate=self.srPositive))

    if (temp3==1):
      self.signalNegative=torch.from_numpy(augment(samples=self.signalNegative.numpy(),sample_rate=self.srNegative))


    self.spectogramAnchor=torchaudio.transforms.Spectrogram(n_fft=320,hop_length=160,win_length=320)(self.signalAnchor)
    self.logSpectogramAnchor=torchaudio.transforms.AmplitudeToDB()(self.spectogramAnchor)

    self.spectogramPositive=torchaudio.transforms.Spectrogram(n_fft=320,hop_length=160,win_length=320)(self.signalPositive)
    self.logSpectogramPositive=torchaudio.transforms.AmplitudeToDB()(self.spectogramPositive)

    self.spectogramNegative=torchaudio.transforms.Spectrogram(n_fft=320,hop_length=160,win_length=320)(self.signalNegative)
    self.logSpectogramNegative=torchaudio.transforms.AmplitudeToDB()(self.spectogramNegative)


    #self.tempImgAnchor=torchvision.transforms.ToPILImage()(self.logSpectogramAnchor)
    #self.tempImgAnchor=self.tempImgAnchor.convert("RGB")
    #self.spectogramAnchorImageTensor=self.vision_transform(self.tempImgAnchor)

    #self.tempImgPositive=torchvision.transforms.ToPILImage()(self.logSpectogramPositive)
    #self.tempImgPositive=self.tempImgPositive.convert("RGB")
    #self.spectogramPositiveImageTensor=self.vision_transform(self.tempImgPositive)

    #self.tempImgNegative=torchvision.transforms.ToPILImage()(self.logSpectogramNegative)
    #self.tempImgNegative=self.tempImgNegative.convert("RGB")
    #self.spectogramNegativeImageTensor=self.vision_transform(self.tempImgNegative)

    return self.logSpectogramAnchor,self.logSpectogramPositive,self.logSpectogramNegative


#Without DALI. For PreTrained VGG

class SpectogramImgTripletTestDataset(Dataset):

  def __init__(self,dataframe,vision_transform=None):
    self.data=dataframe
    self.vision_transform=vision_transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    self.anchor=str(self.data.iloc[idx,0])
    self.positive=self.data.iloc[idx,1]
    self.negative=self.data.iloc[idx,2]

    self.signalAnchor,self.srAnchor=torchaudio.load(self.anchor)
    self.signalPositive,self.srPositive=torchaudio.load(self.positive)
    self.signalNegative,self.srNegative=torchaudio.load(self.negative)

    self.spectogramAnchor=torchaudio.transforms.Spectrogram(n_fft=320,hop_length=160,win_length=320)(self.signalAnchor)
    self.logSpectogramAnchor=torchaudio.transforms.AmplitudeToDB()(self.spectogramAnchor)

    self.spectogramPositive=torchaudio.transforms.Spectrogram(n_fft=320,hop_length=160,win_length=320)(self.signalPositive)
    self.logSpectogramPositive=torchaudio.transforms.AmplitudeToDB()(self.spectogramPositive)

    self.spectogramNegative=torchaudio.transforms.Spectrogram(n_fft=320,hop_length=160,win_length=320)(self.signalNegative)
    self.logSpectogramNegative=torchaudio.transforms.AmplitudeToDB()(self.spectogramNegative)


    #self.tempImgAnchor=torchvision.transforms.ToPILImage()(self.logSpectogramAnchor)
    #self.tempImgAnchor=self.tempImgAnchor.convert("RGB")
    #self.spectogramAnchorImageTensor=self.vision_transform(self.tempImgAnchor)

    #self.tempImgPositive=torchvision.transforms.ToPILImage()(self.logSpectogramPositive)
    #self.tempImgPositive=self.tempImgPositive.convert("RGB")
    #self.spectogramPositiveImageTensor=self.vision_transform(self.tempImgPositive)

    #self.tempImgNegative=torchvision.transforms.ToPILImage()(self.logSpectogramNegative)
    #self.tempImgNegative=self.tempImgNegative.convert("RGB")
    #self.spectogramNegativeImageTensor=self.vision_transform(self.tempImgNegative)

    return self.logSpectogramAnchor,self.logSpectogramPositive,self.logSpectogramNegative

tripletTrainDataset=SpectogramImgTripletTrainDataset(dataframe=dfTripletTrain)
tripletTestDataset=SpectogramImgTripletTestDataset(dataframe=dfTripletTest)
tripletValDataset=SpectogramImgTripletTestDataset(dataframe=dfTripletVal)

tripletTrainLoader=torch.utils.data.DataLoader(tripletTrainDataset,batch_size=4,shuffle=True)
tripletTestLoader=torch.utils.data.DataLoader(tripletTestDataset,batch_size=4,shuffle=True)
tripletValLoader=torch.utils.data.DataLoader(tripletValDataset,batch_size=4,shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.00001)
tripletLoss=nn.TripletMarginLoss() 

def Average(lst): 
    return sum(lst) / len(lst) 

def train_model(model,patience,epochs,avgTrainLoss,avgValidLoss,avgTrainAcc,avgValidAcc):


  trainBatchCount=0
  testBatchCount=0

  avgTrainAcc=avgTrainAcc
  avgValidAcc=avgValidAcc
  trainAcc=[]
  validAcc=[]
  trainLosses=[]
  validLosses=[]
  avgTrainLoss=avgTrainLoss
  avgValidLoss=avgValidLoss

  #early_stopping = EarlyStopping(patience=patience, verbose=True)

  for i in range(epochs):

    print("Epoch:",i)

    model.train()
    print("Training.....")
    #try:
    for batch_idx, (anchor,positive,negative) in enumerate(tripletTrainLoader):


      trainBatchCount=trainBatchCount+1

      anchor=anchor.to(device=device)
      positive=positive.to(device=device)
      negative=negative.to(device=device)

      optimizer.zero_grad()

      anchorEmb=model(anchor,preTrainingFlag=0)
      posEmb=model(positive,preTrainingFlag=0)
      negEmb=model(negative,preTrainingFlag=0)
      
      loss=tripletLoss(anchorEmb,posEmb,negEmb)
      
      loss.backward()

      optimizer.step()

      trainLosses.append(float(loss))

      
      correct=0
      total=0
      total=len(anchor)

      #print("Targets:-",targets)

      #predictions=torch.argmax(scores,dim=1)

      #print("Predictions:-",predictions)

      correct = ( (anchorEmb-posEmb).pow(2).sum(1) < (anchorEmb-negEmb).pow(2).sum(1) ).sum()
      acc=float((correct/float(total))*100)

      trainAcc.append(acc)

      if ((trainBatchCount%200)==0):

        #print("Targets:-",targets)
        #print("Predictions:-",predictions)

        print('Loss: {}  Accuracy: {} %'.format(loss.data, acc))

    #except:
      #continue

    model.eval()
    print("Validating.....")
    #try:
    for anchor,positive,negative in tripletValLoader:

      testBatchCount=testBatchCount+1

      anchor=anchor.to(device=device)
      positive=positive.to(device=device)
      negative=negative.to(device=device)

      anchorEmb=model(anchor,preTrainingFlag=0)
      posEmb=model(positive,preTrainingFlag=0)
      negEmb=model(negative,preTrainingFlag=0)

      loss=tripletLoss(anchorEmb,posEmb,negEmb)

      validLosses.append(float(loss))

      testCorrect=0
      testTotal=0

      #_,predictions=scores.max(1)

      testCorrect = ( (anchorEmb-posEmb).pow(2).sum(1) < (anchorEmb-negEmb).pow(2).sum(1) ).sum()
      testTotal=len(anchor)

      testAcc=float((testCorrect/float(testTotal))*100)

      validAcc.append(testAcc)

      if ((testBatchCount%200)==0):

        #print("Targets:-",targets)
        #print("Predictions:-",predictions)

        print('Loss: {}  Accuracy: {} %'.format(float(loss), testAcc))
    #except:
      #continue
    

    trainLoss=Average(trainLosses)
    validLoss=Average(validLosses)
    avgTrainLoss.append(trainLoss)
    avgValidLoss.append(validLoss)
    tempTrainAcc=Average(trainAcc)
    tempTestAcc=Average(validAcc)
    avgTrainAcc.append(tempTrainAcc)
    avgValidAcc.append(tempTestAcc)

    print("Epoch Number:-",i,"  ","Training Loss:-"," ",trainLoss,"Validation Loss:-"," ",validLoss,"Training Acc:-"," ",tempTrainAcc,"Validation Acc:-"," ",tempTestAcc)

    trainAcc=[]
    ValidAcc=[]
    trainLosses=[]
    validLosses=[]

    #early_stopping(validLoss,model)

    #if early_stopping.early_stop:
      #print("Early Stopping")
      #break

  #model.load_state_dict(torch.load("checkpoint.pt"))

  return model,avgTrainLoss,avgValidLoss,avgTrainAcc,avgValidAcc

model,avgTrainLoss,avgValidLoss,avgTrainAcc,avgValidAcc=train_model(model,3,15,avgTrainLoss,avgValidLoss,avgTrainAcc,avgValidAcc)

loss_train = avgTrainLoss
loss_val = avgValidLoss
epochs = range(1,len(avgTrainLoss)+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss(FineTuning/ResNet-18)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('ResNet-18_fineTuningLoss.png', bbox_inches='tight')
plt.close()

loss_train = avgTrainAcc
loss_val = avgValidAcc
epochs = range(1,len(avgTrainLoss)+1)
plt.plot(epochs, loss_train, 'g', label='Training Acc')
plt.plot(epochs, loss_val, 'b', label='Validation Acc')
plt.title('Training and Validation Accuracy(FineTuning/ResNet-18)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('ResNet-18_fineTuningAcc.png', bbox_inches='tight')
plt.close()


def checkTripletMetrics(loader,model):
  
  completeTargets=[]
  completePreds=[]

  correct=0
  total=0
  model.eval()


  with torch.no_grad():
    for anchor,positive,negative in loader:

      anchor=anchor.to(device=device)
      positive=positive.to(device=device)
      negative=negative.to(device=device)

      anchorEmb=model(anchor,preTrainingFlag=0)
      posEmb=model(positive,preTrainingFlag=0)
      negEmb=model(negative,preTrainingFlag=0)      

      tempPreds=(( (anchorEmb-posEmb).pow(2).sum(1) < (anchorEmb-negEmb).pow(2).sum(1) ).tolist())
      completeTargets.append([1]*len(anchor))
      completePreds.append([int(elem) for elem in tempPreds])


    completeTargetsFlattened=[item for sublist in completeTargets for item in sublist]
    completePredsFlattened=[item for sublist in completePreds for item in sublist]

      #correct += ( (anchorEmb-posEmb).pow(2).sum(1) < (anchorEmb-negEmb).pow(2).sum(1) ).sum()
      #total+=len(anchor)

    cm = ConfusionMatrix(actual_vector=completeTargetsFlattened, predict_vector=completePredsFlattened)
  return cm



tripletCM=checkTripletMetrics(tripletTestLoader,model)

f=open("tripletResults.txt","a")
f.write(str(tripletCM))
f.close()
