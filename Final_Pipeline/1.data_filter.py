import mne
import scipy.io

for i in range(1,10):
    subject_name = f'A0{i}T'
    raw = mne.io.read_raw_gdf(f'/Users/siddharth/Downloads/BCICIV_2a_gdf/{subject_name}.gdf',
                            eog=['EOG-left', 'EOG-central', 'EOG-right'], preload=True)
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    raw.set_eeg_reference()
    channel_rename_dict = {
        'EEG-Fz': 'Fz',
        'EEG-0': 'FC3',
        'EEG-1': 'FC1',
        'EEG-2': 'FCz',
        'EEG-3': 'FC2',
        'EEG-4': 'FC4',
        'EEG-5': 'C5',
        'EEG-C3': 'C3',  # Assuming EEG-C3 is mapped to C1
        'EEG-6': 'C1',
        'EEG-Cz': 'Cz',  # Assuming EEG-Cz is mapped to C2
        'EEG-7': 'C2',
        'EEG-C4' : 'C4',
        'EEG-8': 'C6',
        'EEG-9': 'CP3',
        'EEG-10': 'CP1',
        'EEG-11': 'CPz',
        'EEG-12': 'CP2',
        'EEG-13': 'CP4',
        'EEG-14': 'P1',
        'EEG-Pz': 'Pz',
        'EEG-15': 'P2',
        'EEG-16': 'POz'
    }

    # Rename the channels using the updated dictionary
    raw.rename_channels(channel_rename_dict)

    # print(raw.info['ch_names'])

    montage = mne.channels.make_standard_montage('standard_1020')  # Use 'standard_1020' or custom montage
    raw.set_montage(montage)
    # Step 2: Extract events and create epochs
    events = mne.events_from_annotations(raw)

    print(events)
    epoch = mne.Epochs(raw, events[0], event_id={'769': 7, '770': 8}, tmin=-1.5, tmax=6, on_missing='warn', preload=True)

    labels = epoch.events[:, -1] - 7

    # Step 3: Apply baseline correction
    epoch.apply_baseline((-1.5, 0.0))

    # Step 4: Ensure data is loaded into memory
    epoch.load_data()

    # Step 5: Select signal between t = 3 and t = 6
    epoch_cropped = epoch.crop(tmin=3, tmax=6)
    # Step 6: Apply a Notch filter at 50 Hz
    picks = mne.pick_types(epoch_cropped.info, eeg=True)  # Pick EEG channels

    epoch_cropped._data = mne.filter.notch_filter(epoch_cropped.get_data(),
                                                Fs=epoch_cropped.info['sfreq'],
                                                freqs=50,
                                                picks=picks,
                                                method='spectrum_fit')

    iir_params = {
        'order': 5,  # 5th-order filter
        'ftype': 'butter',  # Butterworth filter
    }

    epoch_cropped._data = mne.filter.filter_data(
        epoch_cropped.get_data(),
        sfreq=epoch_cropped.info['sfreq'],  # Sampling frequency
        l_freq=0.5,  # Low cutoff frequency
        h_freq=48,   # High cutoff frequency
        picks=picks, # Channels to apply the filter to
        iir_params=iir_params,
        # phase='zero',       # Linear-phase filtering
        # pad='edge',         # Padding to minimize edge effects
        method='iir'  # Filter method (IIR is commonly used)
    )



    save_path = '/Users/siddharth/Downloads/epoch_data.mat'
    scipy.io.savemat(save_path, {
        'data': epoch_cropped._data,
        'labels': labels,
        'sfreq': epoch_cropped.info['sfreq'],  # Sampling frequency
        'ch_names': epoch_cropped.info['ch_names'],  # Channel names
        'times': epoch_cropped.times  # Time vector
    })


    save_path = F'/Users/siddharth/Documents/Brainconnectivity/Filtered/filtered_data_subA0{i}T.mat'
    scipy.io.savemat(save_path, {
        'data': epoch_cropped._data,
        'labels': labels,
        'sfreq': epoch_cropped.info['sfreq'],  # Sampling frequency
        'ch_names': epoch_cropped.info['ch_names'],  # Channel names
        'times': epoch_cropped.times  # Time vector
    })