%% Import data from text file
% Script for importing data from the following text file:
%
%    filename: C:\Users\Semih Kaya\Dropbox\2021-2022 Spring Semester\CENG502 - Advanced Deep Learning\Project\Sample Codes\2022-06-26\laplacian_code_Wu\csvResults\groundTruthEigenvectors.csv
%
% Auto-generated by MATLAB on 27-Jun-2022 21:57:46

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 10);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
groundTruthEigenvectors = readtable("groundTruthEigenvectors.csv", opts);

%% Convert to output type
groundTruthEigenvectors = table2array(groundTruthEigenvectors);

%% Clear temporary variables
clear opts