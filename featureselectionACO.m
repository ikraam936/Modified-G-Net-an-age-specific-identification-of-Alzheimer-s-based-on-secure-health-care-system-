clc;
clear all;
close all;

imagespath = imageSet('C:\ADNI\Ver.1\AGE-Specific\AGE_wise_JPG\80-above', 'recursive');
imagecount = 1;

for i = 1:size(imagespath, 2)
    m = size(imagespath(i).ImageLocation, 2);
    temp = imagespath(i).ImageLocation;
    
        folderLabel = imagespath(i).Description;
       
    if contains(folderLabel, 'AD')
        labelChar = 'A';
        labelNum = 1;
    elseif contains(folderLabel, 'CN')
        labelChar = 'B';
        labelNum = 2;
    elseif contains(folderLabel, 'MCI')
        labelChar = 'C';
        labelNum = 3;
    else
        continue; 
    end
    
    
    for j = 1:m
        v{imagecount, 1} = temp{j};      
        v{imagecount, 2} = labelChar;    
        v1(imagecount, 1) = labelNum;    
        imagecount = imagecount + 1;
    end
end

X=v(:,2); 
load('ExtractedFeatures_ModifiedGoogLeNet_80A.mat'); 
inceptionresnetfeatures = double(features); 

ho = 0.3; 
HO = cvpartition(v1,'HoldOut',ho,'Stratify',false);
tau   = 1;
eta   = 1;
alpha = 1.5;
beta  = 1;
rho   = 0.1;
phi   = 0.1;
Nf = 1500;      
N        = 10; 
max_Iter = 100; % Using 5 iterations instead of 100 for testing


fprintf('Shifting Inception-ResNet features to GPU...\n');
X_train_gpu = gpuArray(double(inceptionresnetfeatures)); 


[sFeat_gpu, Nf_out, Sf_gpu, curve_gpu] = ...
    jACO(X_train_gpu, v1, N, max_Iter, ...
         tau, eta, alpha, beta, rho, phi, ...
         Nf, HO);


fprintf('Gathering optimized outputs back to CPU...\n');
sFeat = gather(sFeat_gpu);
Sf    = gather(Sf_gpu);
curve = gather(curve_gpu);
Acc = jKNN(sFeat, v1, HO);

figure;
plot(1:max_Iter, curve, 'LineWidth', 2, 'Color', [0 0.4470 0.7410]);
xlabel('Number of Iterations');
ylabel('Fitness Value'); 
title('BDA Convergence Curve (GPU Accelerated)'); 
grid on;


%__________________________________________________________________________
%||||||||||||||||||||||||||******  250 ******||||||||||||||||||||||||||||||
%**************************************************************************
FusedFeatures0250=sFeat(:,1:250);
FusedFeatures0250(:,size(FusedFeatures0250,2)+1)=cell2mat(X);
save('FusedFeatures80A250.mat','FusedFeatures0250');
FusedFeatures2=FusedFeatures0250;


%__________________________________________________________________________
%||||||||||||||||||||||||||******  500 ******||||||||||||||||||||||||||||||
%**************************************************************************
FusedFeatures0500=sFeat(:,1:500);
FusedFeatures0500(:,size(FusedFeatures0500,2)+1)=cell2mat(X);
save('Fusedfeatures80A500.mat','FusedFeatures0500');
FusedFeatures2=FusedFeatures0500;

%__________________________________________________________________________
%||||||||||||||||||||||||||******  750 ******||||||||||||||||||||||||||||||
%**************************************************************************
FusedFeatures0750=sFeat(:,1:750);
FusedFeatures0750(:,size(FusedFeatures0750,2)+1)=cell2mat(X);
save('FusedFeatures80A750.mat','FusedFeatures0750');
FusedFeatures2=FusedFeatures0750;

%__________________________________________________________________________
%||||||||||||||||||||||||||******  1000 ******||||||||||||||||||||||||||||||
%**************************************************************************
FusedFeatures01000=sFeat(:,1:1000);
FusedFeatures01000(:,size(FusedFeatures01000,2)+1)=cell2mat(X);
save('FusedFeatures80A1000.mat','FusedFeatures01000');
FusedFeatures2=FusedFeatures01000;