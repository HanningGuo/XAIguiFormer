import os
import mne
import numpy as np
import mne_connectivity
from config import get_cfg_defaults
from multiprocessing import Pool


def cal_tbr_con(connectome):
    '''
    construct theta/beta ration connectome
    :param connectome: SpectralConnectivty object, a couple of frequency band connectome
    :return: matrix of shape (n_nodes, n_nodes), theta/beta ratio connectome
    '''

    # calculate theta/beta ratio FC
    theta_con = connectome.get_data('dense')[:, :, 1]
    beta_con = connectome.get_data('dense')[:, :, 8]
    tbr_con = theta_con / beta_con
    # handle division by zero
    tbr_con[np.isinf(tbr_con)] = 0
    tbr_con[np.isnan(tbr_con)] = 0
    return tbr_con


def constructFC(read_fname, save_path, lower_boundary, upper_boundary, methods):
    '''
    construct specific frequency band functional connectivity using mne_connectivity
    :param read_fname: str, the path to access the storage preprocessed EEG data
    :param save_path: str, the directory to save connectome
    :param lower_boundary: tuple or float, frequency band lower boundary
    :param upper_boundary: tuple or float, frequency band upper boundary
    :param methods: list or str, the method of connectome construction
    '''

    # load preprocessed EEG data
    eyes_close = mne.read_epochs(read_fname, proj=False, preload=True)
    # only use eeg channels to estimate connectome
    # eyes_close.pick(picks='eeg')
    eyes_close.pick(picks='eeg', exclude=['A1', 'A2'])   # only for TUAB

    coherence, wpli = [], []
    # coherence (index 0) as node features, wpli (index 1) as edge features
    for single_eyes_close in eyes_close:
        # segment 30s resting state eyes close to 3s short eyes close in order to construct connectome
        eyes_close_long_epoch = mne.io.RawArray(single_eyes_close, eyes_close.info)
        eyes_close_short_epoch = mne.make_fixed_length_epochs(eyes_close_long_epoch, duration=3, preload=True,
                                                              proj=False, id=30)
        # construct connectome
        connectome = mne_connectivity.spectral_connectivity_epochs(
            eyes_close_short_epoch, names=eyes_close.ch_names,
            method=methods, sfreq=eyes_close.info['sfreq'],
            mode='multitaper',
            fmin=lower_boundary, fmax=upper_boundary,
            faverage=True)
        coherence.append(connectome[0])
        wpli.append(connectome[1])

        # calculate theta/beta ratio FC
        # tbr_coh = cal_tbr_con(connectome[0])
        # tbr_wpli = cal_tbr_con(connectome[1])

    # set save path
    __, sub_EID = os.path.split(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_fname_coh = os.path.join(save_path, f'{sub_EID}_EC_coherence.npy')
    save_fname_wpli = os.path.join(save_path, f'{sub_EID}_EC_wpli.npy')

    # save the coherence and wpli lists to disk
    np.save(save_fname_coh, coherence)
    np.save(save_fname_wpli, wpli)


if __name__ == "__main__":

    # load configuration
    cfg = get_cfg_defaults()
    cfg.merge_from_file('../configs/TUAB_preprocess.yaml')
    cfg.freeze()

    # parallel to accelerate the construction of connectome7
    n_workers = cfg.preprocessing.n_workers
    pool = Pool(n_workers)   # a strange bug: can't run it with python console

    # set frequency band range
    lower_boundary = (cfg.connectome.frequency_band.delta[0], cfg.connectome.frequency_band.theta[0], cfg.connectome.frequency_band.low_alpha[0], cfg.connectome.frequency_band.high_alpha[0], cfg.connectome.frequency_band.low_beta[0], cfg.connectome.frequency_band.mid_beta[0], cfg.connectome.frequency_band.high_beta[0], cfg.connectome.frequency_band.gamma[0], cfg.connectome.frequency_band.beta[0])
    upper_boundary = (cfg.connectome.frequency_band.delta[1], cfg.connectome.frequency_band.theta[1], cfg.connectome.frequency_band.low_alpha[1], cfg.connectome.frequency_band.high_alpha[1], cfg.connectome.frequency_band.low_beta[1], cfg.connectome.frequency_band.mid_beta[1], cfg.connectome.frequency_band.high_beta[1], cfg.connectome.frequency_band.gamma[1], cfg.connectome.frequency_band.beta[1])
    # set methods of constructing connectome
    methods = cfg.connectome.methods
    # set load and save directory
    data_dir = cfg.connectome.path.data_dir
    save_dir = cfg.connectome.path.save_dir

    dirs = os.listdir(data_dir)
    for dir in dirs:
        if os.path.isdir(os.path.join(data_dir, dir)):
            sub_EIDs = os.listdir(os.path.join(data_dir, dir))
            for sub_EID in sub_EIDs:
                fname = os.path.join(data_dir, dir, sub_EID, f'{sub_EID}_EC_epo.fif')
                if os.path.exists(fname):
                    # set the path to save connectome
                    save_path = os.path.join(save_dir, dir, sub_EID)
                    # constructFC(fname, save_path, lower_boundary, upper_boundary, methods)
                    pool.apply_async(func=constructFC, args=(fname, save_path, lower_boundary, upper_boundary, methods))

    # parallel construct connectome in TUAB or TDBRAIN biobank
    pool.close()
    pool.join()
