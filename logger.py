import os
import sys
import logging
import datetime
from torch.utils.tensorboard import SummaryWriter


def create_logger(cfg):
    # create the file name for tensorboard writer
    dataset = cfg.dataset
    fname = f'{cfg.model_name}_lr{cfg.train.optimizer.lr}_weightdecay{cfg.train.optimizer.weight_decay}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'

    cfg.defrost()
    cfg.save_fname = fname
    cfg.freeze()

    # create directory if no exist
    os.makedirs(os.path.join(cfg.out_root, 'results', dataset), exist_ok=True)

    # create tensorboard writer
    writer_path = os.path.join(cfg.out_root, 'results', dataset, fname)
    writer = SummaryWriter(writer_path)

    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # logger.setLevel(logging.INFO)

    # create directory if no exist
    os.makedirs(os.path.join(cfg.out_root, 'logs', dataset), exist_ok=True)
    # create file handle and set level
    logger_path = os.path.join(cfg.out_root, 'logs', dataset, f'{fname}.log')
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.DEBUG)
    # fh.setLevel(logging.INFO)

    # create formatter
    fmt = '[%(asctime)s - %(name)s] - (%(filename)s - %(lineno)d): %(levelname)s - %(message)s'
    # add formatter to fh
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    # add fh to logger
    logger.addHandler(fh)

    # create console handlers and set level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    # add formatter to ch
    ch.setFormatter(logging.Formatter(fmt='%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    # add ch to logger
    logger.addHandler(ch)

    return writer, logger
