from utils.visualizer import get_local
get_local.activate()
import os
import time
import copy
import torch
import random
import argparse
import datetime
import numpy as np
from timm.layers import RmsNorm
from data.transform import Mixup
from logger import create_logger
from torch_scatter import scatter
from timm.utils import AverageMeter
from config import get_cfg_defaults
from modules.activation import GeGLU
from utils.eval_metrics import eval_metrics
from models.XAIguiFormer import XAIguiFormer
from torch_geometric.loader import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
from data.EEGBenchmarkDataset import EEGBenchmarkDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # might improve if the benchmarking feature sets True
    torch.backends.cudnn.benchmark = False


def main(cfg):
    # create dataset
    train_dataset = EEGBenchmarkDataset(cfg.root, cfg.dataset, 'train')
    val_dataset = EEGBenchmarkDataset(cfg.root, cfg.dataset, 'val')
    test_dataset = EEGBenchmarkDataset(cfg.root, cfg.dataset, 'test')

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)

    # setup mixup / cutmix
    mixup_active = cfg.aug.mixup_prob > 0
    if mixup_active:
        mixup = Mixup(
            mixup_alpha=cfg.aug.mixup, cutmix_alpha=cfg.aug.cutmix, cutmix_minmax=cfg.aug.cutmix_minmax,
            prob=cfg.aug.mixup_prob, switch_prob=cfg.aug.mixup_switch_prob, mode=cfg.aug.mixup_mode
        )
    else:
        mixup = None

    # frequency band
    freqband = dict(cfg.connectome.frequency_band)
    freqband['beta'] = [freqband['theta'][0] / freqband['beta'][0], freqband['theta'][1] / freqband['beta'][1]]

    # create model
    logger.info(f"Creating model:\n")
    # Reproducibility
    if cfg.seed is not None:
        set_seed(cfg.seed)

    model = XAIguiFormer(
        cfg.model.num_node_feat,
        cfg.model.num_edge_feat,
        cfg.model.dim_node_feat,
        cfg.model.dim_edge_feat,
        cfg.model.num_classes,
        cfg.model.num_gnn_layer,
        cfg.model.num_head,
        cfg.model.num_transformer_layer,
        torch.tensor(list(freqband.values())),
        cfg.model.gnn_type,
        act_func=GeGLU,
        norm=RmsNorm,
        dropout=cfg.model.dropout,
        explainer_type=cfg.model.explainer_type,
        mlp_ratio=cfg.model.mlp_ratio,
        init_values=cfg.model.init_values,
        attn_drop=cfg.model.attn_drop,
        droppath=cfg.model.droppath,
    )
    logger.info(str(model))

    # speed up model using torch.compile
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.optimizer.lr, betas=cfg.train.optimizer.betas,
                                  eps=cfg.train.optimizer.eps, weight_decay=cfg.train.optimizer.weight_decay)

    # create the lr scheduler to adjust learning rate during train
    lr_scheduler = CosineLRScheduler(optimizer, t_initial=cfg.train.epochs * len(train_loader),
                                     lr_min=cfg.train.lr_scheduler.lr_min,
                                     warmup_t=cfg.train.lr_scheduler.warmup_epochs * len(train_loader),
                                     warmup_lr_init=cfg.train.lr_scheduler.warmup_lr)

    # create loss function
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.criterion.smoothing)

    max_perf = 0
    test_bac = test_sensitivity = test_aucpr = test_auroc = test_loss = 0

    logger.info(f"{'-' * 35} Start training {'-' * 35}")
    start_time = time.time()
    for epoch in range(cfg.train.epochs):
        train_bac, train_sensitivity, train_aucpr, train_auroc, train_loss = train(model, criterion, train_loader, optimizer, lr_scheduler, epoch, mixup, device, cfg)

        val_bac, val_sensitivity, val_aucpr, val_auroc, val_loss = validate(model, criterion, val_loader, device, cfg)

        if cfg.dataset == 'TDBRAIN':
            perf = val_sensitivity
        elif cfg.dataset == 'TUAB':
            perf = val_bac
        # update the performance
        if epoch >= cfg.train.epochs/2 and max_perf <= perf:
            max_perf = max(max_perf, perf)
            test_bac, test_sensitivity, test_aucpr, test_auroc, test_loss = validate(model, criterion, test_loader, device, cfg)
            best_model = copy.deepcopy(model.state_dict())

        logger.info(f'Max performance: {max_perf * 100:.2f}%')
        # logger
        logger.info('*' * 80)
        logger.info(f'Epoch: {epoch + 1:03d}\n' +
                    f'train balanced accuracy: {train_bac * 100:.2f}%, sensitivity: {train_sensitivity * 100:.2f}%, aucpr: {train_aucpr:.4f}, auroc: {train_auroc:.4f}, loss: {train_loss:.4f}\n'
                    f'val balanced accuracy: {val_bac * 100:.2f}%, sensitivity: {val_sensitivity * 100:.2f}%, aucpr: {val_aucpr:.4f}, auroc: {val_auroc:.4f}, loss: {val_loss:.4f}\n'
                    f'test balanced accuracy: {test_bac * 100:.2f}%, sensitivity: {test_sensitivity * 100:.2f}%, aucpr: {test_aucpr:.4f}, auroc: {test_auroc:.4f}, loss: {test_loss:.4f}\n'
                    )
        logger.info('*' * 80)
        writer.add_scalars('balanced accuracy',
                           {'train balanced accuracy': train_bac * 100,
                            'val balanced accuracy': val_bac * 100,
                            },
                           epoch)
        writer.add_scalars('loss',
                           {'train loss': train_loss,
                            'val loss': val_loss,
                            },
                           epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time: {total_time_str}')
    logger.info(f"{'-' * 38} Done {'-' * 38}")

    # save the trained model
    fname = cfg.save_fname
    save_path = os.path.join(cfg.out_root, 'results', cfg.dataset, fname, f'{fname}.pt')
    torch.save(best_model, save_path)


def train(model, criterion, data_loader, optimizer, lr_scheduler, epoch, mixup, device, cfg):
    model.train()
    num_steps = len(data_loader)

    loss_meter, bac_meter, sensitivity_meter, aucpr_meter, auroc_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    start = time.time()
    for batch, data in enumerate(data_loader):
        get_local.clear()
        data = data.to(device)

        if mixup is not None:
            y_true = data.y
            data = mixup(data)

        optimizer.zero_grad()
        out = model(data)
        if len(out) > 1:
            loss = (1 - cfg.train.criterion.alpha) * criterion(out[0], data.y) + cfg.train.criterion.alpha * criterion(out[1], data.y)
        else:
            loss = criterion(out[0], data.y)
        loss.backward()
        optimizer.step()

        # measure metrics and record loss
        bac, sensitivity, aucpr, auroc = eval_metrics(out[-1], y_true, cfg.model.num_classes, device)

        loss_meter.update(loss.item(), data.y.size(0))
        bac_meter.update(bac, data.y.size(0))
        sensitivity_meter.update(sensitivity, data.y.size(0))
        aucpr_meter.update(aucpr, data.y.size(0))
        auroc_meter.update(auroc, data.y.size(0))

        # print train information
        if batch % cfg.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Train: [{epoch + 1}/{cfg.train.epochs}][{batch + 1}/{num_steps}]\t' +
                f'lr {lr:.6f}\t wd {wd:.4f}\t' +
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t' +
                f'balanced accuracy {bac * 100:.2f}%\t' +
                f'mem {memory_used:.0f}MB'
            )

        # update scheduler per batch
        lr_scheduler.step(epoch * num_steps + batch)

    total_bac = bac_meter.avg
    epoch_time = time.time() - start

    logger.info('-' * 60)
    logger.info(
        f"EPOCH {epoch + 1}: train balanced accuracy {total_bac * 100:.2f}% train takes {datetime.timedelta(seconds=int(epoch_time))}"
    )
    logger.info('-' * 60)
    return bac_meter.avg, sensitivity_meter.avg, aucpr_meter.avg, auroc_meter.avg, loss_meter.avg


@torch.no_grad()
def validate(model, criterion, data_loader, device, cfg):
    model.eval()

    start = time.time()
    eids, outs, epoch_y = [], [], []
    for batch, data in enumerate(data_loader):
        get_local.clear()
        eids.extend(data.eid)
        data.to(device)
        outs.append(model(data))
        epoch_y.append(data.y)
    out = torch.cat([torch.stack(single_out) for single_out in outs], dim=1)
    epoch_y = torch.cat(epoch_y, dim=0)

    # group the same subject in order to ensemble their performance
    unique_eids = set(eids)
    sub_group = torch.zeros(len(eids), 1, dtype=torch.int64)
    count = 0
    for eid in unique_eids:
        idx = [index for index, item in enumerate(eids) if item == eid]
        sub_group[idx] = count
        count += 1
    sub_group = sub_group.to(device)

    # ensemble the output probability by mean
    out = scatter(out, sub_group.unsqueeze(0), dim=1, reduce='mean')
    target = scatter(epoch_y.unsqueeze(1), sub_group, dim=0, reduce='mean').squeeze()

    # measure performance and record loss
    if len(out) > 1:
        loss = (1 - cfg.train.criterion.alpha) * criterion(out[0], target) + cfg.train.criterion.alpha * criterion(out[1], target)
    else:
        loss = criterion(out[0], target)

    bac, sensitivity, aucpr, auroc = eval_metrics(out[-1], target, cfg.model.num_classes, device)

    # measure elapsed time
    total_time = time.time() - start

    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    logger.info('#' * 60)
    logger.info(
        f'Test: \n' +
        f'time {total_time:.3f}\t' +
        f'loss {loss:.4f}\t' +
        f'balanced accuracy {bac * 100:.2f}%\t' +
        f'mem {memory_used:.0f}MB'
    )
    logger.info('#' * 60)

    return bac, sensitivity, aucpr, auroc, loss


if __name__ == '__main__':

    # get the absolute path of the current script
    current_script_path = os.path.abspath(__file__)
    # get the directory of the current script
    current_script_directory = os.path.dirname(current_script_path)
    # set the working directory to the directory of the current script
    os.chdir(current_script_directory)

    # get the dataset from command line in order to train the model more flexible
    def check_dataset(dataset):
        if dataset in ['TDBRAIN', 'TUAB']:
            return dataset
        else:
            raise argparse.ArgumentTypeError("Dataset must be TDBRAIN or TUAB")

    parser = argparse.ArgumentParser('XAIguiFormer training and evaluation script', add_help=False)
    parser.add_argument('--dataset', type=check_dataset, default='TDBRAIN', help='dataset name')
    args = parser.parse_args()

    # load configuration
    cfg = get_cfg_defaults()
    if args.dataset == 'TDBRAIN':
        cfg.merge_from_file('configs/TDBRAIN_model.yaml')
    elif args.dataset == 'TUAB':
        cfg.merge_from_file('configs/TUAB_model.yaml')
    cfg.freeze()

    # create tensorboard writer and logger
    writer, logger = create_logger(cfg)
    logger.info(cfg.dump())

    main(cfg)
