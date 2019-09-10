# MATLAB_DeepToolbox
matlab deeptoolbox usage
```bash
.
├── 1. Load and Explore Image Data
├── 2. Define Network Architecture
├── 3. Specify Training Options
├── 4. Train Network Using Training Data
├── 5. Classify Validation Images and Compute Accuracy
```

## 1. Load and Explore Image Data
```
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
```

## 2. Define Network Architecture

```
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
```

## 3. Specify Training Options

```
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
```


## 4. Train Network Using Training Data
```
net = trainNetwork(imdsTrain,layers,options);
```
<p align='center'>
<img src="Readme/train_epoch.png" width="600"/> 
</p>

## 5. Classify Validation Images and Compute Accuracy
```
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
```

accuracy = 0.9924

