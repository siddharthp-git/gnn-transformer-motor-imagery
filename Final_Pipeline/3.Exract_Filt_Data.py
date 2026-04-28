import numpy as np
import mne
import h5py
from scipy.stats import spearmanr

def extract_filtered_eeg(i):
    """ Load, preprocess EEG data, and extract connectivity features. """

    raw = mne.io.read_raw_gdf(f'/Users/siddharth/Documents/Brainconnectivity/BCICIV_2a_gdf/A0{i}T.gdf',
                              eog=['EOG-left', 'EOG-central', 'EOG-right'], preload=True)
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    raw.set_eeg_reference()

    # 🔹 Rename EEG channels for consistency
    channel_rename_dict = {
        'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
        'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4',
        'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
        'EEG-14': 'P1', 'EEG-Pz': 'Pz', 'EEG-15': 'P2', 'EEG-16': 'POz'
    }
    raw.rename_channels(channel_rename_dict)
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))

    # 🔹 Extract events
    events, event_id = mne.events_from_annotations(raw)
    epoch = mne.Epochs(raw, events, event_id={'769': 7, '770': 8}, tmin=-1.5, tmax=6, preload=True)

    # 🔹 Extract labels & apply baseline correction
    labels = epoch.events[:, -1] - 7
    labels = np.array(labels)  # Ensure labels are NumPy array
    epoch.apply_baseline((-1.5, 0.0))

    # 🔹 Crop signal from t = 3s to t = 6s
    epoch_cropped = epoch.crop(tmin=3, tmax=6)

    # 🔹 Load filtered EEG data from .mat file
    file_path = f'/Users/siddharth/Documents/Brainconnectivity/filtered_Artifacts_removed/filt_PreProc_EEG_ICA_filt_sub{i}.mat'
    
    def recursive_get_data(group, file):
        """ Recursively extract HDF5 data. """
        data = {}
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Group):
                data[key] = recursive_get_data(item, file)
            elif isinstance(item, h5py.Dataset):
                data[key] = item[:]
            elif isinstance(item, h5py.Reference):
                ref_obj = file[item]
                if isinstance(ref_obj, h5py.Dataset):
                    data[key] = ref_obj[:]
                elif isinstance(ref_obj, h5py.Group):
                    data[key] = recursive_get_data(ref_obj, file)
        return data

    with h5py.File(file_path, 'r') as f:
        all_data = recursive_get_data(f, f)

    eeg_data = []
    for key in all_data.get('#refs#', {}):
        ref_group = all_data['#refs#'][key]
        if ref_group.shape[0] == 751:
            eeg_data.append(ref_group)

    eeg_data = np.stack(eeg_data, axis=0).reshape(144, 22, 751)  # Adjust shape if needed

    return eeg_data, labels, epoch['769'].average()


for i in range(2, 10):
    eeg_data, labels, info = extract_filtered_eeg(i)
    save_path = f'/Users/siddharth/Documents/Brainconnectivity/Filt/eeg_data_sub{i}.mat'
    savemat(save_path, {'eeg_data': eeg_data, 'labels': labels})
    print(f'Saved {save_path}')