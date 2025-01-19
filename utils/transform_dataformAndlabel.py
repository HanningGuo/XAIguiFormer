import numpy as np
import os.path as osp
import os
from constructFC import cal_tbr_con
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import custom_read_raw_edf


# setting the source and target paths according to dataset
dataset = 'TUAB'
label_path = '/path/to/TDBRAIN/labels_with_split.csv'
# demographics_path = '/path/to/TDBRAIN/TDBRAIN_participants_V2.csv'
demographics_path = '/path/to/TUAB/v3.0.1/edf'
source_root = f'/path/to/{dataset}/data'
target_root = f'/path/to/{dataset}/data/dataset'
connectome_dir = osp.join(source_root, 'connectome')
target_dir = osp.join(target_root, dataset, 'raw')

# load the labels and split
if dataset == 'TUAB':
    labels = pd.DataFrame(columns=['EID', 'ABNORMAL', 'NORMAL', 'SPLIT'])
    labels['ABNORMAL'] = labels['ABNORMAL'].astype(int)
    labels['NORMAL'] = labels['NORMAL'].astype(int)
    for split_dir in os.listdir(connectome_dir):
        if osp.isdir(osp.join(connectome_dir, split_dir)):
            for type_dir in os.listdir(osp.join(connectome_dir, split_dir)):
                if osp.isdir(osp.join(connectome_dir, split_dir, type_dir)):
                    for sub_EID in os.listdir(osp.join(connectome_dir, split_dir, type_dir)):
                        if os.path.exists(osp.join(connectome_dir, split_dir, type_dir, sub_EID, f'{sub_EID}_EC_coherence.npy')):
                            abnormal = int(type_dir == 'abnormal')
                            normal = int(type_dir == 'normal')
                            label = {'EID': sub_EID, 'ABNORMAL': abnormal, 'NORMAL': normal, 'SPLIT': split_dir}
                            labels = labels.append(label, ignore_index=True)

    EID = labels[labels['SPLIT'] == 'train']['EID'].tolist()
    EID_eval = labels[labels['SPLIT'] == 'eval']['EID'].tolist()
    y = labels[labels['SPLIT'] == 'train'][['ABNORMAL', 'NORMAL']].to_numpy()

    EID_train, EID_val, y_train, y_val = train_test_split(EID, y, test_size=0.1, stratify=y)

    # indication of split dataset
    for EID in EID_train:
        index = labels[labels['EID'] == EID].index[0]
        labels.loc[index, 'SPLIT'] = 'train'

    for EID in EID_val:
        index = labels[labels['EID'] == EID].index[0]
        labels.loc[index, 'SPLIT'] = 'val'

    for EID in EID_eval:
        index = labels[labels['EID'] == EID].index[0]
        labels.loc[index, 'SPLIT'] = 'test'
else:
    labels = pd.read_csv(label_path)


# --- TUAB ---
age_mapping = {'aaaaanpr_s001_t002': 90., 'aaaaaiae_s001_t000': 90., 'aaaaamcr_s001_t000': 95.,
               'aaaaammv_s001_t001': 92., 'aaaaapmq_s001_t001': 91., 'aaaaappi_s001_t000': 91.,
               'aaaaappk_s001_t001': 91., 'aaaaamjo_s001_t000': 96., 'aaaaanha_s001_t000': 90.,
               'aaaaaosa_s001_t000': 90., 'aaaaaoun_s001_t000': 90., 'aaaaamli_s001_t001': 95.}
# transform connectome (mne_connectivity object) to .npy format
for split_dir in os.listdir(connectome_dir):
    if osp.isdir(osp.join(connectome_dir, split_dir)):
        for type_dir in os.listdir(osp.join(connectome_dir, split_dir)):
            if osp.isdir(os.path.join(connectome_dir, split_dir, type_dir)):
                for sub_EID in os.listdir(osp.join(connectome_dir, split_dir, type_dir)):
                    fname_coh = osp.join(connectome_dir, split_dir, type_dir, sub_EID, f'{sub_EID}_EC_coherence.npy')
                    fname_wpli = osp.join(connectome_dir, split_dir, type_dir, sub_EID, f'{sub_EID}_EC_wpli.npy')
                    fname_eeg = osp.join(demographics_path, split_dir, type_dir, '01_tcp_ar', f'{sub_EID}.edf')
                    if os.path.exists(fname_coh):
                        cohs = np.load(fname_coh, allow_pickle=True)
                        wplis = np.load(fname_wpli, allow_pickle=True)
                        coh_list, wpli_list = [], []
                        for coh, wpli in zip(cohs, wplis):
                            coh_tbr = cal_tbr_con(coh)
                            coh_con = coh.get_data(output='dense')
                            coh_con[:, :, 8] = coh_tbr
                            coh_con = coh_con + np.transpose(coh_con, (1, 0, 2))
                            coh_list.append(coh_con)

                            wpli_tbr = cal_tbr_con(wpli)
                            wpli_con = wpli.get_data(output='dense')
                            wpli_con[:, :, 8] = wpli_tbr
                            wpli_con = wpli_con + np.transpose(wpli_con, (1, 0, 2))
                            wpli_list.append(wpli_con)

                        demographic_info = []
                        EEG_raw = custom_read_raw_edf(fname_eeg, "standard_1020", type_dir)
                        age = EEG_raw.info['subject_info']['age']
                        if 0 < age < 100:
                            demographic_info.append(age)
                        else:
                            demographic_info.append(age_mapping[sub_EID])
                            # raise ValueError('Age invalid! Please check!')

                        # convert to 0 = Male, 1 = Female
                        gender = EEG_raw.info['subject_info']['sex'] - 1
                        if gender == 0 or gender == 1:
                            demographic_info.append(gender)
                        else:
                            raise ValueError('Gender invalid! Please check!')

                        demographic_info = np.array(demographic_info).reshape(2, 1)

                        y = labels[labels['EID'] == sub_EID][['ABNORMAL', 'NORMAL']].to_numpy()

                        split = labels[labels['EID'] == sub_EID]['SPLIT'].to_string(index=False)
                        save_path = osp.join(target_dir, split, sub_EID)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        np.save(osp.join(save_path, f'{sub_EID}_EC_coherence.npy'), coh_list)
                        np.save(osp.join(save_path, f'{sub_EID}_EC_wpli.npy'), wpli_list)
                        np.save(osp.join(save_path, f'{sub_EID}_EC_demographics.npy'), demographic_info)
                        np.save(osp.join(save_path, f'{sub_EID}_EC_label.npy'), y)


# --- TDBRAIN ---
# transform demographics to .npy format
# demographics = pd.read_csv(demographics_path)
# for type_dir in os.listdir(connectome_dir):
#     if osp.isdir(os.path.join(connectome_dir, type_dir)):
#         for sub_EID in os.listdir(osp.join(connectome_dir, type_dir)):
#             if osp.isdir(os.path.join(connectome_dir, type_dir, sub_EID)):
#                 demographic_info = []
#
#                 query_EID = sub_EID.split("_")[0]
#                 session = int(sub_EID.split("_")[1].split("-")[1])
#
#                 age = demographics[(demographics['participants_ID'] == query_EID) & (demographics['sessID'] == session)]['age'].to_numpy().astype(float)
#                 if 0 < age < 100:
#                     demographic_info.append(age)
#                 else:
#                     raise ValueError('Age invalid! Please check!')
#
#                 # (origin) 1 = Male, 0 = Female,
#                 # however, convert to 0 = Male, 1 = Female in order to keep in line with HBN biobank
#                 gender = demographics[(demographics['participants_ID'] == query_EID) & (demographics['sessID'] == session)]['gender'].to_numpy().astype(int)
#                 # convert to 0 = Male, 1 = Female
#                 gender = (gender + 1) % 2
#                 if gender == 0 or gender == 1:
#                     demographic_info.append(gender)
#                 else:
#                     raise ValueError('Gender invalid! Please check!')
#
#                 demographic_info = np.array(demographic_info)
#
#                 split = labels[labels['EID'] == sub_EID]['SPLIT'].to_string(index=False)
#                 np.save(osp.join(target_dir, split, sub_EID, f'{sub_EID}_EC_demographics.npy'), demographic_info)
#
# # transform label to .npy format
# for type_dir in os.listdir(connectome_dir):
#     if osp.isdir(os.path.join(connectome_dir, type_dir)):
#         for sub_EID in os.listdir(osp.join(connectome_dir, type_dir)):
#             if osp.isdir(os.path.join(connectome_dir, type_dir, sub_EID)):
#                 y = labels[labels['EID'] == sub_EID][['ADHD', 'MDD', 'OCD']].to_numpy()
#                 split = labels[labels['EID'] == sub_EID]['SPLIT'].to_string(index=False)
#                 np.save(osp.join(target_dir, split, sub_EID, f'{sub_EID}_EC_label.npy'), y)
#
# # transform connectome (mne_connectivity object) to .npy format
# for type_dir in os.listdir(connectome_dir):
#     if osp.isdir(osp.join(connectome_dir, type_dir)):
#         for sub_EID in os.listdir(osp.join(connectome_dir, type_dir)):
#             if osp.isdir(os.path.join(connectome_dir, type_dir, sub_EID)):
#                 fname_coh = osp.join(connectome_dir, type_dir, sub_EID, f'{sub_EID}_EC_coherence.npy')
#                 fname_wpli = osp.join(connectome_dir, type_dir, sub_EID, f'{sub_EID}_EC_wpli.npy')
#                 if os.path.exists(fname_coh):
#                     cohs = np.load(fname_coh, allow_pickle=True)
#                     wplis = np.load(fname_wpli, allow_pickle=True)
#                     coh_list, wpli_list = [], []
#                     for coh, wpli in zip(cohs, wplis):
#                         coh_tbr = cal_tbr_con(coh)
#                         coh_con = coh.get_data(output='dense')
#                         coh_con[:, :, 8] = coh_tbr
#                         coh_con = coh_con + np.transpose(coh_con, (1, 0, 2))
#                         coh_list.append(coh_con)
#
#                         wpli_tbr = cal_tbr_con(wpli)
#                         wpli_con = wpli.get_data(output='dense')
#                         wpli_con[:, :, 8] = wpli_tbr
#                         wpli_con = wpli_con + np.transpose(wpli_con, (1, 0, 2))
#                         wpli_list.append(wpli_con)
#
#                     split = labels[labels['EID'] == sub_EID]['SPLIT'].to_string(index=False)
#                     np.save(osp.join(target_dir, split, sub_EID, f'{sub_EID}_EC_coherence.npy'), coh_list)
#                     np.save(osp.join(target_dir, split, sub_EID, f'{sub_EID}_EC_wpli.npy'), wpli_list)
