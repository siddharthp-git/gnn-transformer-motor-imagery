clear; close all;

addpath(genpath('*')); % Local path to your fieldtrip toolbox

data_path = '*'; % Local path to your data
save_path = '*'; % Local path where you want to save your pre-processed data

mat_files = dir([data_path,'*.mat']);

for sub = 1:numel(mat_files)
    %% Reading raw MEG signals and temporal filtering
    load([mat_files(idx).folder,'\',mat_files(idx).name]);
    cfg                     = [];
    cfg.channel             = 'MEG'; % 'MEG': For all MEG channels; 'MEGMAG': For magnetometers; 'MEGGRAD': For gradiometer
    cfg.continuous          = 'yes';
    cfg.demean              = 'yes';
    cfg.lpfilter            = 'yes'; % Lowpass Filter
    cfg.lpfreq              = 100;   % Cut-off frequency in Hz
    MEG_data = ft_preprocessing(cfg,dataMAT);

    %% Rejection of SQUID Jump Artifacts
    cfg = [];
    [~,artifact_jump] = ft_artifact_jump(cfg,MEG_data);

    cfg                           = [];
    cfg.artfctdef.reject          = 'partial';
    cfg.artfctdef.jump.artifact   = artifact_jump;
    Filt_MEG_Data = ft_rejectartifact(cfg,MEG_data);

    %% Combine gradiometers from same brain locations (Uncomment this part when analysing Gradiometers)
    cfg = [];
    cfg.method = 'sum'; % Combine the gradiometers using Pythagoras theorem
    Filt_MEG_Data = ft_combineplanar(cfg,Filt_MEG_Data);

    %% Save pre-processed data
    save([save_path,'PreProcessed_',mat_files(sub).name],'Filt_MEG_Data','-v7.3','-nocompression');
end
