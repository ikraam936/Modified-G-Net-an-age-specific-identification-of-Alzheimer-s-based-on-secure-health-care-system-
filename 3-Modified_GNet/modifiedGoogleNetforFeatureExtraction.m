%GoogleNet used for feature extraction on fMRI from ADNI-4 and AD from ADNI-3 on 17-Nov-2025
clear all;
clc;
close all;

% -------------------------------------------------------------------------
% Load Image Dataset
% -------------------------------------------------------------------------
imds = imageDatastore('C:\ADNI_4\JPG\80-above', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% -------------------------------------------------------------------------
% Load Pretrained GoogLeNet
% -------------------------------------------------------------------------
net = googlenet;
lgraph = layerGraph(net);

% -------------------------------------------------------------------------
% Replace 5x5 Convolution Layers with Two 3x3 Convolutions
% -------------------------------------------------------------------------
layers = lgraph.Layers;
connections = lgraph.Connections;

for i = 1:numel(layers)
    if isa(layers(i), 'nnet.cnn.layer.Convolution2DLayer') && all(layers(i).FilterSize == [5 5])
        oldConv = layers(i);
        nameOld = oldConv.Name;
        numFilters = oldConv.NumFilters;
        numChannels = oldConv.NumChannels;

        % Initialize weights and bias properly
        sz = [3 3 numChannels numFilters];
        weights1 = randn(sz, 'single') * sqrt(2 / prod(sz(1:3)));
        bias1 = zeros([1 1 numFilters], 'single');

        sz2 = [3 3 numFilters numFilters];
        weights2 = randn(sz2, 'single') * sqrt(2 / prod(sz2(1:3)));
        bias2 = zeros([1 1 numFilters], 'single');

        % Create new 3x3 blocks with initialization
        conv1 = convolution2dLayer(3, numFilters, 'Padding', 'same', 'Name', [nameOld '_3x3a']);
        conv1.Weights = weights1;
        conv1.Bias = bias1;

        relu1 = reluLayer('Name', [nameOld '_relu1']);

        conv2 = convolution2dLayer(3, numFilters, 'Padding', 'same', 'Name', [nameOld '_3x3b']);
        conv2.Weights = weights2;
        conv2.Bias = bias2;

        relu2 = reluLayer('Name', [nameOld '_relu2']);

        % Replace old layer with the new sequence
        newBlock = [
            conv1
            relu1
            conv2
            relu2
        ];

        lgraph = replaceLayer(lgraph, nameOld, newBlock);
    end
end

% -------------------------------------------------------------------------
% Assemble the Modified Network
% -------------------------------------------------------------------------
modifiedNet = assembleNetwork(lgraph);

% -------------------------------------------------------------------------
% Display Structure
% -------------------------------------------------------------------------
analyzeNetwork(modifiedNet);

% -------------------------------------------------------------------------
% Prepare Data for Feature Extraction
% -------------------------------------------------------------------------
inputSize = modifiedNet.Layers(1).InputSize;
augimds = augmentedImageDatastore(inputSize(1:2), imds);

% -------------------------------------------------------------------------
% Extract Features
% -------------------------------------------------------------------------
featureLayer = 'inception_5b-output';
features = activations(modifiedNet, augimds, featureLayer, 'OutputAs', 'rows');

% -------------------------------------------------------------------------
% Convert to single precision (reduce memory usage)
% -------------------------------------------------------------------------
features = single(features);


% -------------------------------------------------------------------------
% Save Extracted Features
% -------------------------------------------------------------------------
YTrain = imds.Labels;  % labels
save('fMRI_ExtractedFeatures_ModifiedGoogLeNet_80above.mat', 'features', 'YTrain', '-v7.3');

disp('✔ Feature extraction completed.');
disp(['Total images processed: ' num2str(size(features,1))]);
disp(['Feature length per image: ' num2str(size(features,2))]);
