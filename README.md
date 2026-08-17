# Modified-G-Net-an-age-specific-identification-of-Alzheimer-s-based-on-secure-health-care-system-
This repository contains the full end-to-end pipeline for preprocessing, augmenting, extracting features, and classifying fMRI data from the ADNI-4 dataset. 



Instructions for Reviewers

To replicate our study results, please complete the following steps:

Upload/Run Scripts:- Execute the scripts sequentially from Phase 1 through Phase 6.



Repository Structure & Pipeline Flow

-:  Phase 1: Data Preprocessing
-Location:- Folder '1-datapreprocessing'
-Script:- 'preprocessing.py'
-Description:- Converts raw, time-averaged 3D NIfTI ('.nii') fMRI data into standardized 2D JPG images compatible with Deep Learning backbones. Input and output directories must be specified by dataset groups.
-Example Paths Configuration:-
  '''python
  inputDir = 'C:\ADNI\ADNI-4-NII\'
  outputDir = 'C:\ADNI\ADNI-4-JPG\'
  '''

-:  Phase 2: Data Augmentation
-Location:- Folder '2-DCGAN Augmentation'
-Script:- 'dCGAN' (MATLAB Script)
-Description:- Trains a Deep Convolutional Generative Adversarial Network (DCGAN) to generate synthetic 2D medical image slices from the 3D NIfTI neuroimaging data to resolve class imbalances. Configurable by target sample counts and age brackets.
-Example Paths Configuration:-
  '''matlab
  inputDir = 'C:\ADNI\ADNI-4\test\nii';
  outputDir = 'C:\ADNI\ADNI-4\test\aug';
  '''

-:  Phase 3: Deep Feature Extraction
-Location:- Folder '3-modified_GNet'
-Scripts:- 
  1. 'modifiedGoogleNetforFeatureExtraction.m'
  2. 'featureextractionResNet50.m'
  3. 'featureExtractionVGG16.m'
-Description:- Extracts deep spatial features from the preprocessed JPG brain slices using three different deep learning backbones.

-:  Phase 4: Feature Selection & Optimization
-Location:- Folder '4-ACO Feature Selection' *(Note: Please verify folder index number)*
-Script:- 'ACO_Optimization.m' (MATLAB Script)
-Description:- Implements the Ant Colony Optimization (ACO) algorithm to pick the most relevant feature subsets, reducing dimensionality and training times. Updated per age cohort.
-Example Paths Configuration (e.g., Age 80+ Cohort):-
  '''matlab
  imageFolder = 'C:\ADNI_4\JPG\80-above';
  featuresFile = 'fMRI_ExtractedFeatures_ModifiedGoogLeNet_80above.mat';
  % Saves a 1000-feature subset as: 'ACO_FusedFeatures1000_80A.mat'
  '''

-:  Phase 5: Classification
-Location:- Folder '6-Classification on extracted features'
-Script:- 'Classification on extracted features'
-Description:- Executes the final classification task (e.g., AD vs. Normal Controls) using the optimized, fused feature matrices.

-:  Phase 6: Evaluation Metrics & Findings
-Location:- Folder '7-Evaluation'
-Script:- 'KPI from Confusion matrix'
-Description:- Computes essential Key Performance Indicators (KPIs) including Accuracy, Sensitivity, Specificity, and F1-score directly from model output confusion matrices.
-Outputs Folder:- '8-results'
  * Contains empirical metrics inside 'results.xlsx' (or '.mat') 
  * Contains graphical visualizations inside the file 'confusion matrices'.
