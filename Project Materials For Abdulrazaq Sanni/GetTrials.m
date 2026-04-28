clear; close all;

data_path = 'C:\Users\USER\Desktop\EEGData\data\';
save_path = 'C:\Users\USER\Desktop\EEGData\dataNEW\';

mat_files = dir([data_path,'*.mat']);
numel(mat_files)

for idx = 1:numel(mat_files)
    file      = mat_files(idx).name(1:end-4);
    subject   = mat_files(idx).name(1:6);
    disp(['Analysing ',subject])

    %% Reading EEG data
    load([data_path,mat_files(idx).name]);

    %% Select data
    cfg = [];
    cfg.trials = 41:120;
    dataNEW = ft_selectdata(cfg, data);
    
    %% Save new data
    save([save_path,file,'.mat'],'dataNEW','-v7.3','-nocompression')
end