import os
import mne
import numpy as np
import mne_icalabel
from config import get_cfg_defaults
from pyprep.prep_pipeline import PrepPipeline


def custom_read_raw_brainvision(fname, montage, phenotype, preload=True):
    '''
    read the raw EEG data from BrainVision file format

    :param fname: string, raw EEG BrainVision format path
    :param montage: Digmontage, electrode locations
    :param phenotype: string, subject phenotypic information and ID
    :param preload: bool, whether or not preload data
    :return: EEG raw object
    '''

    # read BrainVision file format
    EEG_raw = mne.io.read_raw_brainvision(fname, preload=preload)
    # set channel type according to the TDBRAIN paper
    channel_type = {'VPVA': 'eog', 'VNVB': 'eog',
                    'HPHL': 'eog', 'HNHR': 'eog',
                    'Erbs': 'ecg', 'OrbOcc': 'eog',
                    'Mass': 'emg'}
    EEG_raw.set_channel_types(channel_type)
    # set montage
    EEG_raw.set_montage(montage)
    # set virtual resting state event and event dictionary
    # in order to segment the whole resting state time series to 30s epochs
    EEG_raw.custom_events = mne.make_fixed_length_events(EEG_raw, id=30, start=0, stop=120, duration=40, overlap=20)
    EEG_raw.custom_event_id = {'eyes close': 30}
    # add phenotypic information
    EEG_raw.info['description'] = phenotype

    return EEG_raw


def custom_read_raw_edf(fname, montage, phenotype, preload=True):
    '''
    read the raw EEG data from edf file format

    :param fname: string, raw EEG edf format path
    :param montage: Digmontage, electrode locations
    :param phenotype: string, subject phenotypic information and ID
    :param preload: bool, whether or not preload data
    :return: EEG raw object
    '''

    # read edf file format
    EEG_raw = mne.io.read_raw_edf(fname, preload=preload)
    # rename special names to standard names of the international 10/20 system
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'Fz', 'Cz', 'Pz']
    ch_mapping = {
        'EEG FP1-REF': 'Fp1', 'EEG FP2-REF': 'Fp2', 'EEG F3-REF': 'F3', 'EEG F4-REF': 'F4',
        'EEG C3-REF': 'C3', 'EEG C4-REF': 'C4', 'EEG P3-REF': 'P3', 'EEG P4-REF': 'P4',
        'EEG O1-REF': 'O1', 'EEG O2-REF': 'O2', 'EEG F7-REF': 'F7', 'EEG F8-REF': 'F8',
        'EEG T3-REF': 'T3', 'EEG T4-REF': 'T4', 'EEG T5-REF': 'T5', 'EEG T6-REF': 'T6',
        'EEG A1-REF': 'A1', 'EEG A2-REF': 'A2', 'EEG FZ-REF': 'Fz', 'EEG CZ-REF': 'Cz',
        'EEG PZ-REF': 'Pz'
    }
    EEG_raw.rename_channels(ch_mapping)
    # pick 21 common channels across the TUAB dataset and reorder those channel with specific order
    EEG_raw.reorder_channels(ch_names)
    # set montage
    EEG_raw.set_montage(montage)
    # set virtual resting state event and event dictionary
    # in order to segment the whole resting state time series to 30s epochs
    # as well as discarding the first 60s and the last 60s time points
    EEG_raw.crop(tmin=55, tmax=EEG_raw.n_times / EEG_raw.info['sfreq'] - 55)
    EEG_raw.custom_events = mne.make_fixed_length_events(EEG_raw, id=30, duration=40, overlap=10)
    EEG_raw.custom_event_id = {'eyes close': 30} # not sure if eyes close, but just keep consistent with other dataset event id
    # add phenotypic information
    EEG_raw.info['description'] = phenotype

    return EEG_raw


def preprocessing(EEG_raw, config, verbose=True):
    '''
    preprocessing function for EEG raw data

    :param EEG_raw: EEG raw object
    :param config: dict, contain necessary parameter in the preprocessing
    :param verbose: bool, indicate some preprocessing information
    :return: preprocessed EEG epochs
    '''

    n_jobs = config.preprocessing.n_jobs
    n_step = 0

    # Step 1: find and remove bad channels
    n_step = n_step + 1
    if verbose:
        print('###############################\n')
        print(f'Step {n_step}: Start detecting bad channels...\n')
        print('###############################\n')
    prep_params = {"ref_chs": "eeg", "reref_chs": "eeg",
                   "line_freqs": np.arange(config.preprocessing.line_frequency, EEG_raw.info['sfreq'] / 2, config.preprocessing.line_frequency)}
    filter_kwargs = {"method": "fir", "n_jobs": n_jobs}
    # detect bad channels
    noisy_detector = PrepPipeline(EEG_raw, prep_params, EEG_raw.get_montage(),
                                  channel_wise=True, filter_kwargs=filter_kwargs)
    noisy_detector.fit()
    # remove bad channels
    EEG_raw.info['bads'] = noisy_detector.interpolated_channels

    # Step 2: FIR bandpass filter
    n_step = n_step + 1
    if verbose:
        print('###############################\n')
        print(f'Step {n_step}: Start filter...\n')
        print('###############################\n')
    EEG_bandpass = EEG_raw.filter(l_freq=config.preprocessing.l_freq, h_freq=config.preprocessing.h_freq, n_jobs=n_jobs)

    # Step 3: segment continuous data into eyes close epochs and downsample to 250 Hz
    n_step = n_step + 1
    if verbose:
        print('###############################\n')
        print(f'Step {n_step}: Start segmenting continuous rs data into eyes close epochs...\n')
        print('###############################\n')

    picked_events_onset = EEG_bandpass.custom_events

    # discard fist five seconds and last five seconds because switching task might cause signal instability
    if abs(EEG_bandpass.info['sfreq'] - 500) < 1e-6 or abs(EEG_bandpass.info['sfreq'] - 512) < 1e-6:
        EEG_eyes_close = mne.Epochs(EEG_bandpass, picked_events_onset, {'eyes close': 30}, baseline=None, tmin=5, tmax=35,
                                preload=True, reject=None, proj=False, decim=2)  # downsample to 250 Hz
    else:
        EEG_eyes_close = mne.Epochs(EEG_bandpass, picked_events_onset, {'eyes close': 30}, baseline=None, tmin=5, tmax=35,
                                preload=True, reject=None, proj=False)  # some subjects' sample frequency is 250 Hz

    # Step 4: repair artifact
    n_step = n_step + 1
    if verbose:
        print('###############################\n')
        print(f'Step {n_step}: Start repairing artifact...\n')
        print('###############################\n')
    # fit ICA
    ica = mne.preprocessing.ICA(n_components=config.preprocessing.ica.n_components,
                                method=config.preprocessing.ica.method,
                                fit_params=dict(config.preprocessing.ica.fit_params))
    ica.fit(EEG_eyes_close)
    ica.exclude = []
    # find which ICs match the artifact pattern
    # comprising brain, muscle artifact, eye blink, heart beat,
    # line noise, channel noise and other
    ic_info = mne_icalabel.label_components(EEG_eyes_close, ica, method='iclabel')
    # extract artifact indices
    ic_probability, ic_labels = ic_info['y_pred_proba'], ic_info["labels"]
    ica.exclude = [idx for idx, label in enumerate(ic_labels) if label not in ["brain", "other"] and ic_probability[idx] >= config.preprocessing.ica.ic_label_threshold]
    print(f"Excluding these ICA components: {np.array(ic_labels)[ica.exclude]}")
    # repair ocular, muscle artifact, channel and line noise
    EEG_ica = ica.apply(EEG_eyes_close)

    # Step 5: interpolate bad channels
    n_step = n_step + 1
    if verbose:
        print('###############################\n')
        print(f'Step {n_step}: Start interpolating bad channels...\n')
        print('###############################\n')
    EEG_interp = EEG_ica.interpolate_bads()

    # Step 6: set average reference
    n_step = n_step + 1
    if verbose:
        print('###############################\n')
        print(f'Step {n_step}: Start setting average reference...\n')
        print('###############################\n')
    return EEG_interp.set_eeg_reference(ref_channels='average', ch_type='eeg')


def preprocessingPipeline(fname, montage, phenotype, config, save_path):
    '''
    preprocessing pipeline for EEG raw data in TUAB and TDBRAIN biobank

    :param fname: string, data path
    :param montage: Digmontage, electrode locations
    :param phenotype: string, subject phenotypic information and ID
    :param config: dict, contain necessary parameter in the preprocessing
    :param save_path: string, a path to save preprocessed eyes close epoch
    '''
    # load raw EEG data
    if config.dataset == 'TDBRAIN':
        EEG_raw = custom_read_raw_brainvision(fname, montage, phenotype)
    elif config.dataset == 'TUAB':
        EEG_raw = custom_read_raw_edf(fname, montage, phenotype)

    if EEG_raw.info['description'] != 'bad':
        # preprocess the raw EEG data if the raw data is not labeled as bad
        EEG_prep = preprocessing(EEG_raw, config, verbose=False)
        # if no such directory, create new directory
        if not os.path.exists(save_path):
            dir, __ = os.path.split(save_path)
            os.makedirs(dir)
        # save preprocessed eyes close epoch to specified path
        EEG_prep.save(save_path, fmt='double', overwrite=True)
    else:
        print(f'\n({phenotype}) The subject quality is bad\n')


if __name__ == "__main__":
    # load preprocessed configuration
    cfg = get_cfg_defaults()
    cfg.merge_from_file('../configs/TUAB_preprocess.yaml')
    cfg.freeze()

    # parallel to accelerate the preprocessing pipeline
    # n_workers = cfg.preprocessing.n_workers
    # pool = Pool(n_workers)   # a strange bug: cann't run it with python console

    # set the electrode locations, dataset and save path
    f_eleloc = cfg.preprocessing.path.electrode_locations
    data_dir = cfg.preprocessing.path.data_dir
    save_dir = cfg.preprocessing.path.save_dir

    dirs = os.listdir(data_dir)
    for dir in dirs:
        if os.path.isdir(os.path.join(data_dir, dir)):
            sub_EIDs = os.listdir(os.path.join(data_dir, dir))
            for sub_EID in sub_EIDs:
                if cfg.dataset == 'TDBRAIN':
                    fname = os.path.join(data_dir, dir, sub_EID, f'{sub_EID}_task-restEC_eeg.vhdr')
                    montage = mne.channels.read_custom_montage(f_eleloc, coord_frame=None)
                elif cfg.dataset == 'TUAB':
                    sub_EID = sub_EID.split('.')[0]
                    fname = os.path.join(data_dir, dir, f'{sub_EID}.edf')
                    montage = "standard_1020"
                else:
                    raise ValueError('The dataset doesn\'t exist, please choose \'HBN\', \'TUAB\' or \' TDBRAIN\'!')

                # set the path to save preprocessed eyes close epoch
                save_path = os.path.join(save_dir, dir, sub_EID, f'{sub_EID}_EC_epo.fif')
                if os.path.exists(fname) and not os.path.exists(save_path):
                    preprocessingPipeline(fname, montage, os.path.join(dir, sub_EID), cfg, save_path)
                    # pool.apply_async(func=preprocessingPipeline, args=(fname, montage, os.path.join(dir, sub_EID), config, save_path))
                else:
                    print(f'{sub_EID} haved been done!\n')

    # parallel preprocess all subjects in TUA or TDBRAIN biobank
    # pool.close()
    # pool.join()
