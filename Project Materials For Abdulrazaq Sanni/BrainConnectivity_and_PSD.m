clear; close all;

data_path = 'C:\Users\USER\Desktop\EEGData\dataNEW\';

mat_files = dir([data_path,'*.mat']);
numel(mat_files)

freqBand = 'Alpha';

switch freqBand
    case 'Alpha'
        freqRange = [8,13];
        save_path_BCM = 'C:\Users\USER\Desktop\EEGData\Alpha\';
        save_path_PSD = 'C:\Users\USER\Desktop\EEGData\Alpha\PSD\';
    case 'Beta'
        freqRange = [13,30];
        save_path_BCM = 'C:\Users\USER\Desktop\EEGData\Beta\';
        save_path_PSD = 'C:\Users\USER\Desktop\EEGData\Beta\PSD\';
    case 'Delta'
        freqRange = [0.5,4];
        save_path_BCM = 'C:\Users\USER\Desktop\EEGData\Delta\';
        save_path_PSD = 'C:\Users\USER\Desktop\EEGData\Delta\PSD\';
    case 'Theta'
        freqRange = [4,7];
        save_path_BCM = 'C:\Users\USER\Desktop\EEGData\Theta\';
        save_path_PSD = 'C:\Users\USER\Desktop\EEGData\Theta\PSD\';
    case 'Gamma'
        freqRange = [30,45];
        save_path_BCM = 'C:\Users\USER\Desktop\EEGData\Gamma\';
        save_path_PSD = 'C:\Users\USER\Desktop\EEGData\Gamma\PSD\';
end

for idx = 1:numel(mat_files)
    file      = mat_files(idx).name(1:end-4);
    subject   = mat_files(idx).name(1:6);
    disp(['Analysing ',subject])

    %% Reading EEG data
    load([data_path,mat_files(idx).name]);

    unique_labels = unique(dataNEW.trialinfo);

    for i = 1:numel(unique_labels)

        %% Frequency analysis
        cfg            = [];
        cfg.method     = 'mtmfft';
        cfg.taper      = 'hanning';
        cfg.output     = 'fourier';
        cfg.keeptrials = 'yes';
        cfg.foilim     = freqRange;
        cfg.trials  = dataNEW.trialinfo == unique_labels(i);
        freq        = ft_freqanalysis(cfg, dataNEW);

        %% Brain Connectivity matrix
        cfg            = [];
        cfg.method  = 'coh';
        cfg.complex = 'imag';
        BCM         = ft_connectivityanalysis(cfg, freq);
        AvgBCM = mean(BCM.cohspctrm,3); 
        %% SaveBrain Connectivity matrix
        save([save_path_BCM,file,num2str(unique_labels(i)),'_BCM_',freqBand,'.mat'],'AvgBCM','-v7.3','-nocompression')

        %% PSD analysis
        cfg            = [];
        cfg.method     = 'mtmfft';
        cfg.taper      = 'hanning';
        cfg.output     = 'pow';
        cfg.keeptrials = 'no';
        cfg.foilim     = freqRange;
        cfg.trials  = dataNEW.trialinfo == unique_labels(i);
        psd        = ft_freqanalysis(cfg, dataNEW);
        avgPSD = mean(psd.powspctrm, 2);
        %% Save PSD data
        save([save_path_PSD,file,num2str(unique_labels(i)),'_PSD_',freqBand,'.mat'],'avgPSD','-v7.3','-nocompression')

    end
    clearvars -except data_path save_path_BCM save_path_PSD mat_files freqRange freqBand;
end