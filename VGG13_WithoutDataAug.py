from pycm import *
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
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader


pd.set_option('display.max_colwidth', -1)

dfClassification=pd.read_csv("VastAiCompleteAudioSamples_Half.csv",index_col=0)
dfTriplet=pd.read_csv("VastAiTriplets_Half.csv",index_col=0)
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


freqLabels=torch.tensor(dfClassificationTrain['Label'].value_counts().sort_index(),dtype=torch.double)
weightClass=freqLabels/freqLabels.sum()
weightClass= 1/weightClass
weightClass=(weightClass).tolist()
sampleWeights=[weightClass[i] for i in dfClassificationTrain['Label']]
trainSampler=WeightedRandomSampler(sampleWeights,len(dfClassificationTrain))


class SpectogramImgClassificationDataset(Dataset):

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

classificationTrainDataset=SpectogramImgClassificationDataset(dataframe=dfClassificationTrain)
classificationTestDataset=SpectogramImgClassificationDataset(dataframe=dfClassificationTest)
classificationValDataset=SpectogramImgClassificationDataset(dataframe=dfClassificationVal)

classificationTrainLoader=torch.utils.data.DataLoader(classificationTrainDataset,batch_size=8,sampler=trainSampler)
classificationTestLoader=torch.utils.data.DataLoader(classificationTestDataset,batch_size=8,shuffle=True)
classificationValLoader=torch.utils.data.DataLoader(classificationValDataset,batch_size=8,shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VGG=[64,64,"M",128,128,"M",256,256,"M",512,512,"M"]

class CustomizedVGG(nn.Module):
  def __init__(self,inChannels=1,embSize=512,noClasses=1251):
    super(CustomizedVGG,self).__init__()
    self.inChannels=inChannels
    self.embSize=embSize
    self.noClasses=noClasses
    self.convLayers=self.createConvLayers(arch=VGG)

    self.commonFC1=nn.Linear(92160,4096)
    #self.commonFC2=nn.Linear(16384,4096)
    self.softmaxFC1=nn.Linear(4096,self.noClasses)
    self.tripletFC1=nn.Linear(4096,2048)
    self.tripletFC2=nn.Linear(2048,self.embSize)

  def forward(self,input,preTrainingFlag=0):
    convOutput=self.convLayers(input)
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

  def createConvLayers(self,arch):
    layers=[]
    inChannels=self.inChannels

    for x in arch:
      if (type(x)==int):
        out_channels=x

        layers+=[nn.Conv2d(in_channels=inChannels,out_channels=out_channels,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                 nn.ReLU()]
        inChannels=x
      else:
        layers+=[nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]

    return nn.Sequential(*layers)


model=CustomizedVGG(noClasses=763)
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
    for data,targets in classificationTestLoader:

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
plt.title('Training and Validation loss(Pre-Training)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('preTrainingLoss.png', bbox_inches='tight')
#plt.show()

loss_train = avgTrainAcc
loss_val = avgValidAcc
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training Acc')
plt.plot(epochs, loss_val, 'b', label='Validation Acc')
plt.title('Training and Validation Accuracy(Pre-Training)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('preTrainingAcc.png', bbox_inches='tight')

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

print(softmaxCM)



print("=================================================================================================================================")
print("Fine-Tuning")
print("=================================================================================================================================")
#Without DALI. For PreTrained VGG

from torch.utils.data import Dataset, DataLoader

class SpectogramImgTripletDataset(Dataset):

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


tripletTrainDataset=SpectogramImgTripletDataset(dataframe=dfTripletTrain)
tripletTestDataset=SpectogramImgTripletDataset(dataframe=dfTripletTest)
tripletValDataset=SpectogramImgTripletDataset(dataframe=dfTripletVal)

tripletTrainLoader=torch.utils.data.DataLoader(tripletTrainDataset,batch_size=8,shuffle=True)
tripletTestLoader=torch.utils.data.DataLoader(tripletTestDataset,batch_size=8,shuffle=True)
tripletValLoader=torch.utils.data.DataLoader(tripletValDataset,batch_size=8,shuffle=True)

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
    try:
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

    except:
      continue

    model.eval()
    print("Validating.....")
    try:
      for anchor,positive,negative in tripletTestLoader:

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
    except:
      continue
    

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
plt.title('Training and Validation loss(FineTuning)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('fineTuningLoss.png', bbox_inches='tight')

loss_train = avgTrainAcc
loss_val = avgValidAcc
epochs = range(1,len(avgTrainLoss)+1)
plt.plot(epochs, loss_train, 'g', label='Training Acc')
plt.plot(epochs, loss_val, 'b', label='Validation Acc')
plt.title('Training and Validation Accuracy(FineTuning)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('fineTuningAcc.png', bbox_inches='tight')

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

      tempPreds=(( (temp1-temp2).pow(2).sum(1) < (temp1-temp3).pow(2).sum(1) ).tolist())
      completeTargets.append([1]*len(anchor))
      completePreds.append([int(elem) for elem in tempPreds])

    completeTargetsFlattened=[item for sublist in completeTargets for item in sublist]
    completePredsFlattened=[item for sublist in completePreds for item in sublist]

      #correct += ( (anchorEmb-posEmb).pow(2).sum(1) < (anchorEmb-negEmb).pow(2).sum(1) ).sum()
      #total+=len(anchor)

    cm = ConfusionMatrix(actual_vector=completeTargetsFlattened, predict_vector=completePredsFlattened)
  return cm


tripletCM=checkTripletMetrics(tripletTestLoader,model)

print(tripletCM)

