clear variables; close all; clc;

% All scripts and csv files must be located in the same folder

loadGroundTruthEigenvectorsToWorkspace;
loadStateRepresentationsToWorkspace;

d = size(stateRepresentations, 2);

stateRepresentationsNormalized = (-1)*ones(size(stateRepresentations));

for i = 1:d
    stateRepresentationsNormalized(:, i) = stateRepresentations(:, i)/norm(stateRepresentations(:, i));
end

similarity = abs(dot(stateRepresentationsNormalized, groundTruthEigenvectors));
disp(['Similarity = ', num2str(mean(similarity))]);

similarityMatrix = abs(cosineSimilarity(groundTruthEigenvectors', stateRepresentationsNormalized'));