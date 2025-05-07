function [] = extract_hctsa_features(data_id, data_dir, feature_set, ops_file, use_parallel)
% extract_hctsa_features:
% Converts a .mat matrix of time series into hctsa format, initializes metadata,
% and computes time-series features using the hctsa framework.
%
% Inputs:
%   - data_id       : Identifier used to construct input/output file names (e.g., 'sub01')
%   - data_dir      : Directory containing the input .mat files and where outputs will be saved
%   - feature_set   : A string specifying the feature set to use:
%                    * 'all'     — extract the full hctsa feature set (~7700 features)
%                    * 'RSRD44'  — extract the 44 features used in our study
%   - ops_file      : Path to the operations file (required only if feature_set ≠ 'all')
%   - use_parallel  : Logical flag indicating whether to use parallel processing

% Example:
% extract_hctsa_features('sub01', './data', 'RSRD44', './ops/ops_RSRD44.txt', true)

% =========================================================================
% IMPORTANT: Please update the following line to match the path to your
% local hctsa installation. This script was developed and tested using
% hctsa version 1.06.
% =========================================================================
% addpath(genpath('/share/user_data/public/MatlabToolbox/hctsa'));

% -------------------------------------------------------------------------
% Load time-series matrix
% -------------------------------------------------------------------------
input_matfile = fullfile(data_dir, strcat(data_id, '_INP.mat'));
if ~isfile(input_matfile)
    error('Input file not found: %s', input_matfile);
end

data = load(input_matfile);
ts_data = double(data.data);
[~, num_series] = size(ts_data);

% Prepare required hctsa input cell arrays
timeSeriesData = cell(1, num_series);
labels = cell(1, num_series);
keywords = cell(1, num_series);  % Optional metadata; left empty

for i = 1:num_series
    timeSeriesData{i} = ts_data(:, i);
    labels{i} = data.labels(i, :);
end

% -------------------------------------------------------------------------
% Save *_INIT.mat file for hctsa initialization
% -------------------------------------------------------------------------
init_file = fullfile(data_dir, strcat(data_id, '_INIT.mat'));
save(init_file, 'timeSeriesData', 'labels', 'keywords');
fprintf('[INFO] Saved INIT file: %s\n', init_file);

% -------------------------------------------------------------------------
% Initialize hctsa output structure
% -------------------------------------------------------------------------
output_file = fullfile(data_dir, strcat(data_id, feature_set, '_OUT.mat'));

if strcmpi(feature_set, 'all') % hctsa full set
    TS_Init(init_file, 'INP_mops.txt', 'INP_ops.txt', false, output_file);
    % ref to : https://github.com/benfulcher/hctsa/blob/main/Calculation/TS_Init.m
else
    if ~isfile(ops_file)
        error('Specified operations file not found: %s', ops_file);
    end
    TS_Init(init_file, 'INP_mops.txt', ops_file, false, output_file);
end
fprintf('[INFO] Initialized OUT file: %s\n', output_file);

% -------------------------------------------------------------------------
% Compute features using hctsa
% -------------------------------------------------------------------------
TS_Compute(use_parallel, [], [], [], output_file, false);
% ref to : https://github.com/benfulcher/hctsa/blob/main/Calculation/TS_Compute.m

fprintf('[INFO] Feature extraction complete: %s\n', output_file);

end
