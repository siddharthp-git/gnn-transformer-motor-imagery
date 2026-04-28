clear; close all;

freqBand = 'Alpha';

switch freqBand
    case 'Alpha'
        data_path = 'C:\Users\USER\Desktop\EEGData\Alpha\';
    case 'Beta'
        data_path = 'C:\Users\USER\Desktop\EEGData\Beta\';
    case 'Delta'
        data_path = 'C:\Users\USER\Desktop\EEGData\Delta\';
    case 'Theta'
        data_path = 'C:\Users\USER\Desktop\EEGData\Theta\';
    case 'Gamma'
        data_path = 'C:\Users\USER\Desktop\EEGData\Gamma\';
end

mat_files = dir([data_path,'*.mat']);
numel(mat_files)

for idx = 1:numel(mat_files)
    file      = mat_files(idx).name(1:end-4);
    subject   = mat_files(idx).name(1:6);
    disp(['Analysing ',subject])

    %% Reading data
    load([data_path,mat_files(idx).name]);
    
    %% Uncomment node feature to get and run

%     %% 1.Get Efficiency
%     efficiency = efficiency_wei(AvgBCM,2);
% 
%     %% Save Efficiency
%     save_path = [data_path, 'Efficiency\'];
%     save([save_path,file,'.mat'],'efficiency','-v7.3','-nocompression')

%     %% 2.Get Clustering Coefficients
%     clusteringcoef = clustering_coef_wu(AvgBCM);
% 
%     %% Save Clustering Coefficients
%     save_path = [data_path, 'ClusteringCoef\'];
%     save([save_path,file,'.mat'],'clusteringcoef','-v7.3','-nocompression')

%     %% 3.Get Strengths
%     strengths = strengths_und(AvgBCM);
% 
%     %% Save Strengths
%     save_path = [data_path, 'Strength\'];
%     save([save_path,file,'.mat'],'strengths','-v7.3','-nocompression')

%     %% 4.Get Betweenness
%     W = weight_conversion(AvgBCM, 'lengths');
%     betweenness = betweenness_wei(W);
% 
%     %% Save Betweenness
%     save_path = [data_path, 'Betweenness\'];
%     save([save_path,file,'.mat'],'betweenness','-v7.3','-nocompression')

%     %% 5.Get Hjorth Parameters
%     [ACTIVITY, MOBILITY, COMPLEXITY] = hjorth(AvgBCM,0);
% 
%     %% Save Parameters
%     save_path = [data_path, 'Activity\'];
%     save([save_path,file,'.mat'],'ACTIVITY','-v7.3','-nocompression')
% 
%     save_path = [data_path, 'Mobility\'];
%     save([save_path,file,'.mat'],'MOBILITY','-v7.3','-nocompression')
% 
%     save_path = [data_path, 'Complexity\'];
%     save([save_path,file,'.mat'],'COMPLEXITY','-v7.3','-nocompression')

end