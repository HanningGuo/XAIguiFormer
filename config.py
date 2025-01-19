from yacs.config import CfgNode as CN


_C = CN()

########################################################################
# basic settings
########################################################################
# model name
_C.model_name = 'XAIguiFormer'
# dataset name, TUAB or TDBRAIN
_C.dataset = 'TUAB'
# the root path of dataset
_C.root = '<root-path-to-dataset>'
# the output root path of tensorboard writer and logger
_C.out_root = '<output-root-to-save-results>'
# the number of threads/workers for data loading
_C.num_workers = 4
# print frequency
_C.print_freq = 1
# pretrain mode
_C.pretrain = False
# pretrained model
_C.pretrained_model = ''
# Whether fix the running seed to remove randomness
_C.seed = None


########################################################################
# preprocessing settings
########################################################################
_C.preprocessing = CN()
# the number of threads/workers for multiprocess
_C.preprocessing.n_workers = 4
# the number of threads/workers for mne process
_C.preprocessing.n_jobs = None
# high pass frequency
_C.preprocessing.l_freq = 1
# low pass frequency
_C.preprocessing.h_freq = 45
# line frequency, 60 in U.S. or 50 in Europe
_C.preprocessing.line_frequency = 60
# EEG system and dataset path
_C.preprocessing.path = CN()
# EEG system electrode locations
_C.preprocessing.path.electrode_locations = '<file-path-to-elec_loc.csv>'
# raw dataset path
_C.preprocessing.path.data_dir = '<path-to-raw-dataset>'
# save path for preprocessed results
_C.preprocessing.path.save_dir = '<save-path-to-preprocesses-data>'
# ica parameters
_C.preprocessing.ica = CN()
# ica number of components
_C.preprocessing.ica.n_components = 20
# ica method
_C.preprocessing.ica.method = 'infomax'
# the fit parameters of ica method
_C.preprocessing.ica.fit_params = CN()
_C.preprocessing.ica.fit_params.extended = True
# the threshold of the ICALabel to pick up desired artificial component
_C.preprocessing.ica.ic_label_threshold = 0.0

######################################################
# frequency band range and methods of constructing FC
######################################################
_C.connectome = CN()
# frequency band division
_C.connectome.frequency_band = CN()
_C.connectome.frequency_band.delta = [2., 4.]
_C.connectome.frequency_band.theta = [4., 8.]
_C.connectome.frequency_band.low_alpha = [8., 10.]
_C.connectome.frequency_band.high_alpha = [10., 12.]
_C.connectome.frequency_band.low_beta = [12., 18.]
_C.connectome.frequency_band.mid_beta = [18., 21.]
_C.connectome.frequency_band.high_beta = [21., 30.]
_C.connectome.frequency_band.gamma = [30., 45.]
# being used for theta/beta ratio FC
_C.connectome.frequency_band.beta = [12., 30.]
# construction method
_C.connectome.methods = ['coh', 'wpli']
# preprocessed dataset path
_C.connectome.path = CN()
_C.connectome.path.data_dir = '<preprocessed-path-to-preprocessed-data>'
# connectome save path
_C.connectome.path.save_dir = '<save-path-to-connectome>'

######################################################
# model settings
######################################################
_C.model = CN()
# the number of node features
_C.model.num_node_feat = 26
# the number of edge features
_C.model.num_edge_feat = 1
# the number of class
_C.model.num_classes = 3
# the number of head in transformer
_C.model.num_head = 4
# map the original node features to dimension of embedding (d_model)
_C.model.dim_node_feat = 128
# map the original edge features to dimension of embedding
_C.model.dim_edge_feat = 128
# gnn type
_C.model.gnn_type = 'GINEConv'
# the number of gnn layer
_C.model.num_gnn_layer = 4
# the number of transformer layer
_C.model.num_transformer_layer = 4
# explainable artificial intelligence algorithm type
_C.model.explainer_type = 'DeepLift'
# drop out rate
_C.model.dropout = 0.
# the ratio is used to calculate the dimension of feed forward layer
_C.model.mlp_ratio = 4.
# the initial value of layer scale
_C.model.init_values = None
# the attention drop out rate
_C.model.attn_drop = 0.
# the drop path rate
_C.model.droppath = 0.

######################################################
# train settings
######################################################
_C.train = CN()
# batch size
_C.train.batch_size = 64
# the number of epoch
_C.train.epochs = 100

# optimizer
_C.train.optimizer = CN()
# optimizer epsilon
_C.train.optimizer.eps = 1e-8
# optimizer betas
_C.train.optimizer.betas = (0.9, 0.99)
# base learning rate
_C.train.optimizer.lr = 1e-4
# weight decay
_C.train.optimizer.weight_decay = 0.01

# lr scheduler
_C.train.lr_scheduler = CN()
_C.train.lr_scheduler.lr_min = 1e-7
_C.train.lr_scheduler.warmup_epochs = 5
_C.train.lr_scheduler.warmup_lr = 1e-5

# loss function
_C.train.criterion = CN()
_C.train.criterion.smoothing = 0.1
_C.train.criterion.alpha = 1.

######################################################
# augmentation settings
######################################################
_C.aug = CN()
# mixup alpha, mixup enabled if > 0
_C.aug.mixup = 0.2  # large value leads to underfitting
# cutmix alpha, cutmix enabled if > 0
_C.aug.cutmix = 0.2
# cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.aug.cutmix_minmax = None
# probability of performing mixup or cutmix when either/both is enabled
_C.aug.mixup_prob = 1.0
# probability of switching to cutmix when both mixup and cutmix enabled
_C.aug.mixup_switch_prob = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.aug.mixup_mode = 'batch'


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
