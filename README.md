# XAIguiFormer
This repository contains the official PyTorch implementation of XAIguiFormer, as presented in our ICLR 2025 paper [XAIguiFormer: explainable artificial intelligence guided transformer for brain disorder identification](https://openreview.net/forum?id=AD5yx2xq8R).

## Abstract
EEG-based connectomes offer a low-cost and portable method to identify brain disorders using deep learning. With the growing interest in model interpretability and transparency, explainable artificial intelligence (XAI) is widely applied to understand the decision of deep learning models. However, most research focuses solely on interpretability analysis based on the insights from XAI, overlooking XAI’s potential to improve model performance. To bridge this gap, we propose a dynamical-system-inspired architecture, XAI guided transformer (XAIguiFormer), where XAI not only provides explanations but also contributes to enhancing the transformer by refining the originally coarse information in self-attention mechanism to capture more relevant dependency relationships. In order not to damage the connectome’s topological structure, the connectome tokenizer treats the single-band graphs as atomic tokens to generate a sequence in the frequency domain. To address the limitations of conventional positional encoding in understanding the frequency and mitigating the individual differences, we integrate frequency and demographic information into tokens via a rotation matrix, resulting in a richly informative representation. Our experiment demonstrates that XAIguiFormer achieves superior performance over all baseline models. In addition, XAIguiFormer provides valuable interpretability through visualization of the frequency band importance.

![XAIguiFormer](XAIguiFormer.png)

## Environment Set Up
A suitable conda environment named XAIguiFormer can be created and activated with:
```
conda env create -f environment.yaml
conda activate XAIguiFormer
```
or install the required packages step by step:
```
conda create --name XAIguiFormer python=3.10
conda activate XAIguiFormer
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
conda install tensorboard
conda install -c conda-forge einops
conda install pandas
conda install matplotlib
conda install -c conda-forge yacs
conda install pytorch-scatter -c pyg
conda install -c conda-forge -c pytorch -c defaults timm
conda install -c conda-forge torchmetrics
conda install captum -c pytorch
conda install bytecode
```

## Data Preparation
To preprocess the EEG data and construct the connectome, ensure the following libraries (mne, mne_connectivity, mne_icalabel) are installed:
```
conda create -c conda-forge --strict-channel-priority --name=mne mne
conda activate mne
conda install -c conda-forge mne-connectivity
conda install -c conda-forge mne-icalabel
```

### Preprocessing
One could preprocess the raw EEG data by utils/preprocessing. Before running this code, please don't forget to modify the path of EEG data accordingly in the configs/*_preprocess.yaml and load the corresponding YAML file in the preprocessing.py:
```
python utils/preprocessing.py
```

### Construct the connectome
To construct the connectome from preprocessed EEG data, you must configure the data paths correctly in the YAML configuration file:
```
python utils/constructFC.py
```

After completing the EEG data preprocessing and connectome construction, you need to prepare the demographics data and organize all required files into the final directory structure as below. You may refer to utils/transform_dataformAndlabel.py.
```
EEGBenchmarkDataset
├── ...
├── TUAB
|   ├── raw             # preprocessed EEG data
|   |   ├── train       # train dataset
|   |   |     ├── subject_001
|   |   |     |       ├── subject_001_coherence.npy
|   |   |     |       ├── subject_001_wpli.npy
|   |   |     |       ├── subject_001_demographics.npy
|   |   |     |       └── subject_001_label.npy
|   |   |     ├── subject_002
|   |   |     |       ├── subject_002_coherence.npy
|   |   |     |       ├── subject_002_wpli.npy
|   |   |     |       ├── subject_002_demographics.npy
|   |   |     |       └── subject_002_label.npy
|   |   |     └── ...
|   |   ├── val         # val dataset
|   |   |     └── ...
|   |   └── test        # test dataset
|   |         └── ...
|   └── ...
└── ...
```

## Run Experiments
```
python main.py --dataset TDBRAIN/TUAB
```
If you want to change the default hyperparameters, you can update the configuration located in the configs/*_model.yaml.


## Citation
If you find our paper/code useful, please consider citing our work:
```
@inproceedings{
guo2025XAIguiFormer,
title={XAIguiFormer: explainable artificial intelligence guided transformer for brain disorder identification},
author={Hanning Guo, Farah Abdellatif, Yu Fu, N. Jon Shah, Abigail Morrison, Jürgen Dammers},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=AD5yx2xq8R}
}
```
