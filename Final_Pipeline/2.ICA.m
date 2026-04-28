% Load the saved epoch_data.mat file
for i = 1:10
    filePath = sprintf('/Users/siddharth/Documents/Brainconnectivity/filtered/filtered_data_subA0%dT.mat', i);
    load(filePath);

    
    % Ensure 'data' has the expected structure
    [num_trials, num_channels, num_timepoints] = size(data);
    % [ALLEEG, EEG, CURRENTSET] = eeglab;
    
    % Initialize the EEG_Data array
    EEG_Data = cell(1, num_trials);
    
    % Define channel location file path
    chanlocs_path = '/Users/siddharth/Documents/Brainconnectivity/Channel_locs.locs';
    
    for ses = 1:num_trials
        % Extract trial-specific EEG data
        trial_data = squeeze(data(ses, :, :));
        EEG = pop_importdata('data' ,trial_data, 'dataformat','array', 'srate',250,'chanlocs','/Users/siddharth/Documents/Brainconnectivity/Channel_locs.locs');
        % Create new EEG structure
        % EEG = eeg_emptyset();
        % 
        % % Import the trial EEG data
        % EEG.data = trial_data;
        % EEG.srate = sfreq;
        % EEG.nbchan = size(trial_data, 1);
        % EEG.pnts = size(trial_data, 2);
        % EEG.trials = 1;
        % EEG.xmin = 0;
        % EEG.xmax = (EEG.pnts - 1) / EEG.srate;
        % 
        % % Read channel locations
        % EEG = pop_chanedit(EEG, 'load', {chanlocs_path, 'filetype', 'loc'});
        % 
        % % Run basic EEGLAB checks
        % EEG = eeg_checkset(EEG);    
        % Compute rank of data
        % data_rank = rank(EEG.data * EEG.data');
        % disp(['Data rank: ', num2str(data_rank)])
        % Run ICA
        EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1,'pca',21);
        
        % Run ICLabel
        EEG = iclabel(EEG);
        
        % Flag components with high artifact probability
        EEG = pop_icflag(EEG, [0.95 1; 0.95 1; 0.95 1; 0.95 1; 0.95 1; 0.95 1; 0.95 1]);
        
        % Find artifacts
        artifacts = find(EEG.reject.gcompreject > 0);
        
        % Remove artifactual components
        EEG = pop_subcomp(EEG, artifacts);
        % Store processed EEG
        EEG_Data{ses} = EEG.data;  % Store complete EEG structure
    end
    
    % Save the processed data
    fileName = sprintf('filtered_Artifacts_removed/filt_PreProc_EEG_ICA_filt_sub%d.mat', i);
    save(fileName, 'EEG_Data', '-v7.3', '-nocompression');
end