clear all;
clc;
close all;

imds = imageDatastore('C:\ADNI\Ver.1\AGE-Specific\AGE_wise_JPG\61-70', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

numTrainImages = numel(imds.Labels);

%net = inceptionresnetv2;
net = resnet50;
%load TrainedNetwork-Results-BrainTumor4classesNN3.mat
%net = trainedNetwork_3;
 
net.Layers
analyzeNetwork(net)

inputSize = net.Layers(1).InputSize;


augmentedTrainingSet = augmentedImageDatastore(inputSize(1:2),imds);

%layer = 'avg_pool';
layer = 'fc1000';
inceptionresnetfeatures = activations(net,augmentedTrainingSet,layer,'OutputAs','rows');

YTrain = imds.Labels;

save('FeatureExtraction-67-ResNet50.mat','inceptionresnetfeatures');