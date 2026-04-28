% Define the file path
file_path = '/Users/siddharth/Documents/Brainconnectivity/filt/eeg_data_sub';

% List of subjects
subject_count = 9; % Modify according to the number of subjects

for i = 2:subject_count
    % Load EEG data
    file_name = [file_path, num2str(i), '.mat'];  % Example: fil_PreProc_EEG_ICA_filt_sub1.mat
    disp(['Loading file: ', file_name]);

    % Read the data (assuming the structure contains the EEG data)
    eeg_data_struct = load(file_name);
    eeg_data = eeg_data_struct.eeg_data; % Replace 'eeg_data' with the correct variable name if needed

    % Example: Reshape eeg_data if required, assuming it's of shape (144, 22, 751)
    % eeg_data = reshape(eeg_data, [144, 22, 751]);  % Modify this according to actual data structure

    % Extract features

    % Feature 1: Peak amplitude (maximum value per channel)
    peakAmplitude = max(eeg_data, [], 3);
    % disp(size(peakAmplitude));

    % Feature 2: Latency (index of the peak amplitude per channel)
    [~, latency] = max(abs(eeg_data), [], 3);
    % disp(size(latency));

    % Feature 3: Area under the curve (AUC) using trapezoidal integration
    auc = trapz(eeg_data, 3);
    % disp(size(auc));

    % Feature 4: Slope (difference between first and last points in each channel)
    slope = eeg_data(:, :, end) - eeg_data(:, :, 1);
    % disp(size(slope));

    % Feature 5: Peak-to-peak amplitude (difference between max and min values)
    peakToPeakAmplitude = max(eeg_data, [], 3) - min(eeg_data, [], 3);
    % disp(size(peakToPeakAmplitude));

    % Feature 6: Mean absolute amplitude
    meanAbsoluteAmplitude = mean(abs(eeg_data), 3);
    % disp(size(meanAbsoluteAmplitude));
    % Feature 7: Root mean square (RMS) amplitude
    rmsAmplitude = sqrt(mean(eeg_data.^2, 3));
    % disp(size(rmsAmplitude));

    % Feature 8: Standard deviation
    standardDeviation = std(eeg_data, 0, 3);
    % disp(size(standardDeviation));

    % Feature 9: Skewness
    skewnessVal = skewness(eeg_data, 0, 3);
    % disp(size(skewnessVal));

    % Feature 10: Kurtosis
    kurtosisVal = kurtosis(eeg_data, 0, 3);
    % disp(size(kurtosisVal));

    % Feature 11: Zero-crossing rate over kurtosis (ZORK)
    zeroCrossings = sum(diff(eeg_data > 0, 1, 3), 3);
    zork = zeroCrossings ./ kurtosisVal;
    % disp(size(zork));

    % Feature 12: Power Spectral Density (PSD) using Welch's method
    fs = 250;  % Sampling frequency
    window_size = 751;  % Adjust according to your data
    overlap = 128;  % Adjust according to your data
    nfft = 256;  % Number of FFT points
    
    % Ensure that the window size is not larger than the signal length
    n_trials = size(eeg_data, 1);  % Number of trials (rows)
    n_channels = size(eeg_data, 2);  % Number of channels (columns)
    signal_length = size(eeg_data, 3);  % Length of the signal (datapoints per trial)
    
    % If window size is larger than the signal length, adjust it
    if window_size > signal_length
        window_size = signal_length;
        disp('Window size adjusted to signal length');
    end
    
    psdMatrix = zeros(n_trials, n_channels, nfft / 2 + 1);
    
    for ch = 1:n_channels
        for trial = 1:n_trials
            % Extract the current trial's data for the current channel
            trial_data = squeeze(eeg_data(trial, ch, :));
            
            % Compute the power spectral density using pwelch
            [pxx, f] = pwelch(trial_data, window_size, overlap, nfft, fs);
            
            % Store the result in the psd matrix
            psdMatrix(trial, ch, :) = pxx;
        end
    end
    
    % Feature 13: Mean power
    meanPower = mean(psdMatrix, 3);
    
    % Feature 14: Variance of power
    variancePower = var(psdMatrix, 0, 3);
    
    % Feature 15: Skewness of power
    skewnessPower = skewness(psdMatrix, 0, 3);
    
    % Feature 16: Kurtosis of power
    kurtosisPower = kurtosis(psdMatrix, 0, 3);
    
    % Feature 17: Maximum power
    maximumPower = max(psdMatrix, [], 3);
    
    % Feature 18: Spectral entropy
    spectralEntropy = -sum(psdMatrix .* log(psdMatrix + eps), 3) ./ log(size(psdMatrix, 3));
    
    % Combine all the features into one matrix
    Feature = [peakAmplitude, latency, auc, slope, peakToPeakAmplitude, meanAbsoluteAmplitude, ...
        rmsAmplitude, standardDeviation, skewnessVal, kurtosisVal, zork, ...
        meanPower, variancePower, skewnessPower, kurtosisPower, maximumPower, spectralEntropy];
    
    % Save the features
    save_file_name = [file_path, num2str(i), '_features.mat'];
    save(save_file_name, 'Feature', 'peakAmplitude', 'latency', 'auc', 'slope', ...
        'peakToPeakAmplitude', 'meanAbsoluteAmplitude', 'rmsAmplitude', ...
        'standardDeviation', 'skewnessVal', 'kurtosisVal', 'zork', 'meanPower', ...
        'variancePower', 'skewnessPower', 'kurtosisPower', 'maximumPower', ...
        'spectralEntropy');
end
