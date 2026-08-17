clc; clear; close all;

% ---------- Paths & Configuration ----------
inputDir  = 'C:\ADNI\ViT\Nii\ADNI_4_Nii_711';
outputDir = 'C:\ADNI\Ver.1\AGE-Specific\AGE_wise_JPG - original-backup\60-below\ADAUG';

if ~exist(outputDir, 'dir'), mkdir(outputDir); end

% ---------- DCGAN Parameters ----------
imageSize     = 64;       
latentSize    = 100;
batchSize     = 64;       
numEpochs     = 100;      
lrG           = 0.0002;   
lrD           = 0.0002;   

% ---------- Generation Configurations ----------
numSyntheticImages = 13;   
numSlicesPerImage  = 48;   
targetOutputSize   = [224 224]; 
startCaseNumber    = 1;

rng(42); 

% ---------- Read NIfTI Files ----------
niiFiles = dir(fullfile(inputDir, '*.nii'));
numFiles = length(niiFiles);
if numFiles == 0, error('No NIfTI files found.'); end

% ---------- Subject-Level Data Split (70/15/15) ----------
shuffledIndices = randperm(numFiles);
numTrain = floor(0.70 * numFiles);
numVal   = floor(0.15 * numFiles);
numTest  = numFiles - (numTrain + numVal);

trainIndices = shuffledIndices(1:numTrain);
fprintf('Total Files: %d | Train: %d | Val: %d | Test: %d\n', numFiles, numTrain, numVal, numTest);

% ---------- Data Loading ----------
trainingImages = zeros(imageSize, imageSize, 1, numTrain, 'single');
imageCount = 0;

fprintf('Loading training NIfTI files and extracting central slices...\n');
for i = trainIndices
    try
        fileName = fullfile(niiFiles(i).folder, niiFiles(i).name);
        volume = single(niftiread(fileName));
        volume = squeeze(volume);
        
        if ndims(volume) == 2
            image2D = volume;
        elseif ndims(volume) == 3
            midSlice = round(size(volume, 3) / 2);
            image2D = volume(:, :, midSlice);
        elseif ndims(volume) == 4
            midTime = round(size(volume, 4) / 2);
            volume3D = volume(:, :, :, midTime);
            midSlice = round(size(volume3D, 3) / 2);
            image2D = volume3D(:, :, midSlice);
        else
            continue;
        end
        
        image2D(~isfinite(image2D)) = 0;
        
        pLow = prctile(image2D(:), 1);
        pHigh = prctile(image2D(:), 99);
        image2D = max(min(image2D, pHigh), pLow);
        
        minValue = min(image2D(:));
        maxValue = max(image2D(:));
        if maxValue > minValue
            image2D = (image2D - minValue) / (maxValue - minValue);
        else
            image2D = zeros(size(image2D), 'single');
        end
        
        image2D = imresize(image2D, [imageSize, imageSize]);
        imageCount = imageCount + 1;
        trainingImages(:, :, 1, imageCount) = image2D;
    catch
    end
end

trainingImages = trainingImages(:, :, :, 1:imageCount);
trainingImages = (trainingImages * 2) - 1;
trainingImages = max(min(trainingImages, 1), -1);

% ---------- Network Architecture Setup ----------
generatorLayers = [
    imageInputLayer([1 1 latentSize], 'Normalization', 'none')
    transposedConv2dLayer(4, 256, 'Stride', 1, 'Cropping', 0)
    batchNormalizationLayer 
    reluLayer
    transposedConv2dLayer(4, 128, 'Stride', 2, 'Cropping', 'same')
    batchNormalizationLayer
    reluLayer
    transposedConv2dLayer(4, 64, 'Stride', 2, 'Cropping', 'same')
    batchNormalizationLayer
    reluLayer
    transposedConv2dLayer(4, 1, 'Stride', 2, 'Cropping', 'same')
    tanhLayer
];
generator = dlnetwork(generatorLayers);

discriminatorLayers = [
    imageInputLayer([imageSize imageSize 1], 'Normalization', 'none')
    convolution2dLayer(4, 64, 'Stride', 2, 'Padding', 'same')
    leakyReluLayer(0.2)
    convolution2dLayer(4, 128, 'Stride', 2, 'Padding', 'same')
    batchNormalizationLayer 
    leakyReluLayer(0.2)
    convolution2dLayer(4, 256, 'Stride', 2, 'Padding', 'same')
    batchNormalizationLayer
    leakyReluLayer(0.2)
    convolution2dLayer(4, 1, 'Stride', 1, 'Padding', 0)
    sigmoidLayer
];
discriminator = dlnetwork(discriminatorLayers);

avgGradG = []; avgGradSqG = [];
avgGradD = []; avgGradSqD = [];
iteration = 0;
numImages = size(trainingImages, 4);

% ---------- DCGAN Training Loop ----------
fprintf('\nTraining DCGAN (%d epochs, Batch Size %d)...\n', numEpochs, batchSize);
for epoch = 1:numEpochs
    order = randperm(numImages);
    for startIndex = 1:batchSize:numImages
        endIndex = min(startIndex + batchSize - 1, numImages);
        batchIndex = order(startIndex:endIndex);
        
        realImages = trainingImages(:, :, :, batchIndex);
        currentBatchSize = size(realImages, 4);
        if currentBatchSize < 2, continue; end
        
        realImages = dlarray(realImages, 'SSCB');
        noise = dlarray(randn(1, 1, latentSize, currentBatchSize, 'single'), 'SSCB');
        iteration = iteration + 1;
        
        [gradG, gradD, lossG, lossD] = dlfeval(@modelGradients, generator, discriminator, noise, realImages);
        
        [generator, avgGradG, avgGradSqG] = adamupdate(generator, gradG, avgGradG, avgGradSqG, iteration, lrG);
        [discriminator, avgGradD, avgGradSqD] = adamupdate(discriminator, gradD, avgGradD, avgGradSqD, iteration, lrD);
    end
    
    if mod(epoch, 10) == 0 || epoch == 1
        fprintf('Epoch %d/%d | Gen Loss: %.4f | Disc Loss: %.4f\n', ...
            epoch, numEpochs, double(extractdata(lossG)), double(extractdata(lossD)));
    end
end

% ---------- Generating High-Resolution Synthetic Outputs ----------
fprintf('\nGenerating and saving uniform synthetic datasets...\n');
currentCase = startCaseNumber;

for caseIdx = 1:numSyntheticImages
    for sliceIdx = 1:numSlicesPerImage
        noise = dlarray(randn(1, 1, latentSize, 1, 'single'), 'SSCB');
        fakeImage = predict(generator, noise);
        
        imgData = squeeze(extractdata(fakeImage));
        imgNormalized = (imgData + 1) / 2;
        
        % Meets requirements: 224x224 pixels, 0-255 values, 24-bit RGB JPG
        imgResized = imresize(imgNormalized, targetOutputSize);
        imgUint8 = im2uint8(imgResized);
        imgRGB = repmat(imgUint8, [1, 1, 3]);
        
        fileName = sprintf('syn_case_%04d_slice_%02d.jpg', currentCase, sliceIdx);
        imwrite(imgRGB, fullfile(outputDir, fileName), 'jpg');
    end
    currentCase = currentCase + 1;
end
fprintf('Process Complete. Output folder: %s\n', outputDir);

% ---------- Helper Functions Component ----------
function [gradG, gradD, lossG, lossD] = modelGradients(generator, discriminator, noise, realImages)
    fakeImages = forward(generator, noise);
    probReal = forward(discriminator, realImages);
    probFake = forward(discriminator, fakeImages);
    
    % FIXED: Added mean() wrapper around elements to collapse loss to a scalar value
    lossD = mean(-log(probReal + 1e-5) - log(1 - probFake + 1e-5), 'all');
    lossG = mean(-log(probFake + 1e-5), 'all');
    
    gradG = dlgradient(lossG, generator.Learnables);
    gradD = dlgradient(lossD, discriminator.Learnables);
end
