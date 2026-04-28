function [coherency_network_per_frequency,freqoi] = connectivity_coh(data,fs,freq_range,time,complex)

% Input:  data    --> trials x chan x points (3d matrix)
%         complex --> 'abs' : for absolute coherence (power-coherence)
%                     'imag': for imaginary part of coherency (phase-coherence / phase sychrony)
%         fs --> Sampling frequency of the signal
%         freq_range [lowFreq,highFreq] --> Frequency range for which coherency needs to be computed
%         time --> time stamps
% 
% Output: graph --> chan x chan x freq
%
% Note: This function needs Fieldtrip toolbox for 

if numel(size(data)) == 2
    num_trial = 1;
    num_chan = size(data,1);
else
    [num_trial,num_chan,~] = size(data);
end
ft = true;

if ft == true
    % fft using fieldtrip toolbox
    for trl = 1:num_trial
        [fourier_spectrum(trl,:,:),~,freqoi] = ft_specest_mtmfft...
            (squeeze(data(trl,:,:)),time,'freqoi',freq_range(1):freq_range(2),'taper','hanning');
    end
else
    % Bandpass filtering
    for trl = 1:num_trial
        [filtered_data(trl,:,:)] = ft_preproc_bandpassfilter...
            (squeeze(data(trl,:,:)),fs,freq_range);
    end
    
    % fft
    for trl = 1:num_trial
        for chan = 1:num_chan
            fourier_spectrum(trl,chan,:) = fft(squeeze(filtered_data(trl,chan,:)));
        end
    end
end


for sig1 = 1:num_chan
    freqSig1 = squeeze(fourier_spectrum(:,sig1,:));
    for sig2 = 1:num_chan
        freqSig2 = squeeze(fourier_spectrum(:,sig2,:));

        % cross-spectrum
        XSpec = mean(freqSig1.*conj(freqSig2));

        spec1 = mean(freqSig1.*conj(freqSig1));
        spec2 = mean(freqSig2.*conj(freqSig2));

        % coherency
        C = XSpec./sqrt(spec1.*spec2);

        switch complex
            case 'abs'
                C = abs(C);
            case 'imag'
                C = imag(C);
        end

        % Brain Connectivity Network
        coherency_network_per_frequency(sig1,sig2,:) = C;
    end
end