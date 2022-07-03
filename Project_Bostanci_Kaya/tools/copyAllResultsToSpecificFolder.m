clear variables; close all; clc;

environments = ["gridMaze"; "gridRoom"];
losses = ["lossWang"; "lossWu"];
runs = ["run1"; "run2"; "run3"];
runResults = ["run1_results"; "run2_results"; "run3_results"];
constantFolders = fullfile('log', 'visualize_reprs');

selectFoldersFlag = false;
if selectFoldersFlag
    currentFolder = pwd;
    inputPath = uigetdir(currentFolder, 'Select \environments folder to copy files');
    % Sample inputPath:
    % C:\Users\Semih Kaya\Desktop\ForGithub\CENG502-Spring2022\Project_Bostanci_Kaya\environments

    outputPath = uigetdir(currentFolder, 'Select destination folder to copy files');
    % Sample outputPath:
    % C:\Users\Semih Kaya\Desktop\AllResults
else
    inputPath = "C:\Users\Semih Kaya\Desktop\ForGithub\CENG502-Spring2022\Project_Bostanci_Kaya\environments";
    outputPath = "C:\Users\Semih Kaya\Desktop\AllResults";
end

numberOfCopiedFiles = 0;
numberOfExpectedFilesToBeCopied = 12*21;  
for environmentIndex = 1:length(environments)
    for lossIndex = 1:length(losses)
        for runIndex = 1:length(runs)
            inputRunResultsFolder = fullfile(inputPath, environments(environmentIndex), ...
                losses(lossIndex), runs(runIndex), constantFolders);
            % Sample inputRunResultsFolder:
            % C:\Users\Semih Kaya\Desktop\ForGithub\CENG502-Spring2022\Project_Bostanci_Kaya\environments\gridMaze\lossWang\run1\log\visualize_reprs
            outputRunResultsFolder = fullfile(outputPath, environments(environmentIndex), ...
                losses(lossIndex), runResults(runIndex));
            % Sample inputRunResultsFolder:
            % C:\Users\Semih Kaya\Desktop\AllResults\environments\gridMaze\lossWang\run1_results
            cd(inputRunResultsFolder);
            filesToBeCopiedInDirectory = dir;
            % Actually there exists 21 files to be copied, but Matlab finds
            % 23 files.
            for fileIndex = 1:length(filesToBeCopiedInDirectory)
                if (length(filesToBeCopiedInDirectory(fileIndex).name) > 3)
                    copyfile(filesToBeCopiedInDirectory(fileIndex).name, outputRunResultsFolder);
                    numberOfCopiedFiles = numberOfCopiedFiles + 1;
                    disp(strcat(filesToBeCopiedInDirectory(fileIndex).name, " is copied to ", outputRunResultsFolder));
                    disp(" ");
                end
            end
        end
    end
end

disp(" ");
disp(strcat("Total number of expected files to be copied: ", num2str(numberOfExpectedFilesToBeCopied)));
disp(strcat("Total number of files which are copied: ", num2str(numberOfCopiedFiles)));

% Generate a warning message if the number of files which are copied is 
% different than the number of files to be copied
if (numberOfCopiedFiles-numberOfExpectedFilesToBeCopied) == 0
    disp("All files are copied successfully");
else
    warning("Some files may not be copied or extra unexpected files are copied!");
end
disp(" ");