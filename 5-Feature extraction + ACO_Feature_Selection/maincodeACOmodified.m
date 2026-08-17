
clc;
clearvars -except;    % avoid clearing custom functions on some MATLAB setups
close all;

% ---------- Paths ----------
imageFolder = 'C:\ADNI_4\JPG\80-above';
featuresFile = 'fMRI_ExtractedFeatures_ModifiedGoogLeNet_80above.mat';

% ---------- Read images with imageDatastore ----------
imds = imageDatastore(imageFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

numFiles = numel(imds.Files);
if numFiles == 0
    error('No images found in the specified folder: %s', imageFolder);
end

% Preallocate containers
v = cell(numFiles,2);   % {path, labelChar}
v1 = zeros(numFiles,1); % numeric label
cnt = 0;

% Map folder labels to numeric & char labels
for k = 1:numFiles
    imgPath = imds.Files{k};
    folderLabel = char(imds.Labels(k)); % e.g. 'AD', 'MCI', 'CN' (or folder name that contains these)
    
    % Determine label
    if contains(folderLabel, 'AD', 'IgnoreCase', true)
        labelChar = 'A';
        labelNum  = 1;
    elseif contains(folderLabel, 'MCI', 'IgnoreCase', true)
        labelChar = 'M';
        labelNum  = 2;
    elseif contains(folderLabel, 'CN', 'IgnoreCase', true)
        labelChar = 'C';
        labelNum  = 3;
    else
        % skip unknown folders (do not count)
        continue;
    end
    
    cnt = cnt + 1;
    v{cnt,1} = imgPath;
    v{cnt,2} = labelChar;
    v1(cnt,1) = labelNum;
    unique(v1); 
end

% Trim unused preallocated rows (if some files were skipped)
if cnt < numFiles
    v = v(1:cnt,:);
    v1 = v1(1:cnt);
end

% Make sure we have at least two classes
if numel(unique(v1)) < 2
    error('Need at least two classes present in labels. Check folder names and contents.');
end

X = v(:,2);  % cell array of labelChars (A/M/C)

% ---------- Load features ----------
if ~isfile(featuresFile)
    error('Features file not found: %s', featuresFile);
end
S = load(featuresFile);         % load into struct
if isfield(S,'features')
    features = double(S.features);
else
    % try common alternate names
    fld = fieldnames(S);
    % take first numeric matrix-like field
    features = [];
    for i = 1:numel(fld)
        vfld = S.(fld{i});
        if isnumeric(vfld) && ismatrix(vfld)
            features = double(vfld);
            break;
        end
    end
    if isempty(features)
        error('Could not find a numeric "features" matrix inside %s', featuresFile);
    end
end

% ---------- Ensure feature/label row alignment ----------
% Many feature extractors produce features per image in same order as files.
% We check row count and trim/pad as needed (trim: conservative).
nFeatRows = size(features,1);
nLabels   = numel(v1);

if nFeatRows ~= nLabels
    warning('Number of feature rows (%d) does not match number of labels (%d). Trimming to min length.', nFeatRows, nLabels);
    minRows = min(nFeatRows, nLabels);
    features = features(1:minRows, :);
    v1 = v1(1:minRows);
    X = X(1:minRows);
end

% Convert features to double (already done) and check for NaN/Inf
if any(isnan(features(:))) || any(isinf(features(:)))
    error('Features contain NaN or Inf values. Please clean or regenerate features.');
end

% ---------- Hold-out partition ----------
%% ---------- ACO + Top-N Feature Selection ----------

% Desired number of features
desiredNumFeatures = 1000;

% Hold-out partition
ho = 0.3;
HO = cvpartition(v1,'HoldOut',ho,'Stratify',true);

% ACO parameters
N        = 10;   % number of ants
max_Iter = 100;   % iterations

tau   = 1;      % initial pheromone
eta   = 1;      % heuristic info
alpha = 1.5;    % pheromone importance
beta  = 1;      % heuristic importance
rho   = 0.1;    % pheromone evaporation (slower, keeps more features)
phi   = 0.1;    % additional parameter
Nf    = min(desiredNumFeatures, size(features,2)); % max features to select

% ---------- Run ACO ----------
[sFeat, Sf, Nf_out, curve] = jACO(features, v1, N, max_Iter, tau, eta, alpha, beta, rho, phi, Nf, HO);

% Number of features selected
numSelected = size(sFeat,2);
fprintf('Features selected by ACO: %d\n', numSelected);

% ---------- Take top-N features ----------
numTop = min(desiredNumFeatures, numSelected);
sFeatTop = sFeat(:, 1:numTop);   % select top features

% ---------- Fuse with labels ----------
FusedFeatures = sFeatTop;
numRows = size(FusedFeatures,1);
FusedFeatures(:, numTop+1) = v1(1:numRows); % append numeric labels

% ---------- Save ----------
save('ACO_80A_1000.mat','FusedFeatures');

% ---------- Optional: plot convergence curve ----------
if exist('curve','var') && ~isempty(curve)
    figure;
    plot(1:numel(curve), curve, 'LineWidth', 1.5);
    xlabel('Number of Iterations');
    ylabel('Fitness Value');
    title('ACO Convergence Curve');
    grid on;
end

fprintf('ACO + Top-N feature selection complete. %d features fused with labels.\n', size(FusedFeatures,2)-1);


% ---------- Prepare top-100 fused features ----------
numTake = 1000;
if size(sFeat,2) < numTake
    numTake = size(sFeat,2);
    warning('Less than 1000 features selected by ACO. Using %d features.', numTake);
end

FusedFeatures1000 = sFeat(:,1:numTake);

% Convert label characters into numeric column (we already have v1 trimmed)
% Ensure tempX is numeric column vector
tempX = v1;  % already numeric and trimmed above

% Match rows and fuse
minRows = min(size(FusedFeatures1000,1), numel(tempX));
FusedFeatures1000 = FusedFeatures1000(1:minRows,:);
tempX = tempX(1:minRows,1);

% Append labels as last column
FusedFeatures1000(:, size(FusedFeatures1000,2) + 1) = tempX;

% Save result
save('ACO_FusedFeatures1000_80A.mat', 'FusedFeatures1000');

% Also assign to FusedFeatures2 to match your later code usage
FusedFeatures2 = FusedFeatures1000;

% (Optional) Save result structure for classifiers later
% result100 = [];
% save('ACO-result0100.mat', 'result100');

fprintf('Processing complete. %d images processed, %d features kept.\n', minRows, size(FusedFeatures1000,2)-1);

