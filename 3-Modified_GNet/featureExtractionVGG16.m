clear all;
clc;
close all;

% Define your target image repository 
imds = imageDatastore('C:\ADNI\Ver.1\AGE-Specific\AGE_wise_JPG\80-above', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

numTrainImages = numel(imds.Labels);

% Load the pre-trained VGG-16 network
net = vgg16;
 
net.Layers
analyzeNetwork(net)

% Dynamically acquire the target VGG-16 input frame size (224x224x3)
inputSize = net.Layers(1).InputSize;

% Format input image boundaries cleanly to match VGG-16 constraints
augmentedTrainingSet = augmentedImageDatastore(inputSize(1:2),imds);

% Extract deep activation maps from the final fully connected layer before classification
layer = 'fc8';
vgg16features = activations(net,augmentedTrainingSet,layer,'OutputAs','rows');

YTrain = imds.Labels;

% Save final extracted configurations systematically
save('FeatureExtraction-80A-VGG16.mat','vgg16features');
