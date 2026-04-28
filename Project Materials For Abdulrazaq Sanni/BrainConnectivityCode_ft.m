clear; close all;

data_path = ['C:\Users\B00896414\OneDrive - Ulster University\' ...
    'ISRC-CN3 Autumn School 2023 Project Data Repository\EEG Data Associated with Inner Speech\'];

save_path = '*';

mat_files = dir([data_path,'*.mat']);
numel(mat_files)

freqBand = 'Beta';

switch freqBand
    case 'Alpha'
        freqRange = [8,13];
    case 'Beta'
        freqRange = [13,30];
    case 'Delta'
        freqRange = [0.5,4];
    case 'Theta'
        freqRange = [4,7];
    case 'Gamma'
        freqRange = [30,45];
end

for idx = 1:numel(mat_files)
    file      = mat_files(idx).name(1:end-4);
    subject   = mat_files(idx).name(1:6);
    disp(['Analysing ',subject])

    %% Reading EEG data
    load([data_path,mat_files(idx).name]);

    %% Frequency Analysis
    cfg            = [];
    cfg.method     = 'mtmfft';
    cfg.taper      = 'hanning';
    cfg.output     = 'fourier';
    cfg.trials     = 'all';
    cfg.keeptrials = 'yes';
    cfg.foilim     = freqRange;
    freq           = ft_freqanalysis(cfg, data);

    %% Brain Connectivity matrix
    cfg         = [];
    cfg.method  = 'coh';
    cfg.complex = 'imag';
    BCM         = ft_connectivityanalysis(cfg, freq);
    % AvgBCM = mean(BCM.cohspctrm,3); % Uncomment to calculate average BCM across all frequencies
    %% Save
    save([save_path,file,'_BCM_',freqBand,'.mat'],'BCM','-v7.3','-nocompression')

    clearvars -except data_path save_path mat_files freqRange freqBand;
end