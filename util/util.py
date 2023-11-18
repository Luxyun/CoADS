# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler
from torch.utils.data.dataloader import default_collate
import torch_geometric
from torch_geometric.data import Batch


def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.opt == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError("Optimizer not found")
    return optimizer


def get_scheduler(optimizer, args):
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup, eta_min=0)
    elif args.scheduler =='min':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    elif args.scheduler == 'warmcosine':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=args.epochs,cycle_mult=1.0, 
                                                  max_lr=args.lr, min_lr = 0, warmup_steps=args.epochs/10, gamma=1.0)
    elif args.scheduler == 'poly':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9)
    elif args.scheduler is None:
        scheduler=None
    else:
        raise ValueError("Scheduler not found")
    return scheduler

def get_criterion(bag_loss,alpha_surv):
    if bag_loss == 'nll_loss':
        loss_fn = NLLSurvLoss(alpha=alpha_surv)
    elif bag_loss == 'cox_loss':
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError
    return loss_fn


def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

     
def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()
    return l1_reg


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    '''
    :param hazards: output of the network after the sigmoid function
    :param S: survival risk
    :param Y: discrete label
    :param c: event
    :param alpha:
    :param eps:
    :return:
    '''
    c = 1 - c
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  
    S_padded = torch.cat([torch.ones_like(c), S], 1)  
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(
        torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


def _neg_partial_log(hazards,S,Y,c):
    current_batch_len = len(S)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = Y[j] >= Y[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.cuda()
    train_ystatus = torch.FloatTensor(c).cuda()
    theta = S.reshape(-1)
    exp_theta = torch.exp(theta)
    loss = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

    return loss


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


class CoxSurvLoss(object):
    def __call__(self, hazards, S, Y, c,alpha=None):
        return _neg_partial_log(hazards,S,Y,c)


class EarlyStopping:
    def __init__(self, warmup=10, patience=20, stop_epoch=20, verbose=False):
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):
        score = -val_loss
        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)

        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss
        
class Monitor_CIndex:
    def __init__(self, warmup=5, patience=20, stop_epoch=20, verbose=False):
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.valid_cindex_min = -np.Inf

    def __call__(self, epoch, val_cindex, model, ckpt_name: str = 'checkpoint.pt'):
        score = val_cindex

        if epoch < self.warmup:
            pass

        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, ckpt_name)

        elif score < self.best_score:
            self.counter += 1
            # print(f'Early_stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(score, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_cindex, model, ckpt_name):
        if self.verbose:
            print(f'Valid CI increased ({self.valid_cindex_min:.4f} --> {val_cindex:.4f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.valid_cindex_min = val_cindex
       
        
def collate_MIL_survival(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    event_time = np.array([item[2] for item in batch])
    c = torch.FloatTensor([item[3] for item in batch])
    return [img, label, event_time, c]


def collate_MIL_survival_vit(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    grids = torch.cat([item[1] for item in batch], dim=0)
    label = torch.LongTensor([item[2] for item in batch])
    event_time = np.array([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])
    return [img, grids, label, event_time, c]


def collate_MIL_survival_cluster(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    cluster_ids = torch.cat([item[1] for item in batch], dim=0).type(torch.LongTensor)
    label = torch.LongTensor([item[2] for item in batch])
    event_time = np.array([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])
    return [img, cluster_ids, label, event_time, c]


def collate_MIL_survival_graph(batch):
    elem = batch[0]
    elem_type = type(elem)
    transposed = zip(*batch)
    return [samples[0] if isinstance(samples[0], torch_geometric.data.Batch) else default_collate(samples) for samples
            in transposed]

class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)       
        
def get_split_loader(split_dataset, training=False, testing=False, weighted=False, mode='coattn', batch_size=1):
    """
        return either the validation loader or training loader
    """
    if mode == 'graph':
        collate = collate_MIL_survival_graph

    elif mode == 'cluster':
        collate = collate_MIL_survival_cluster

    elif mode == 'path':
        collate = collate_MIL_survival

    elif mode == 'vit' or mode == 'set':
        collate = collate_MIL_survival_vit

    else:
        raise NotImplementedError

    kwargs = {'collate_fn': collate}

    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size,
                                    sampler=WeightedRandomSampler(weights, len(weights)), **kwargs)
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    else:
        ids = np.random.choice(len(split_dataset), int(len(split_dataset) * 0.2), replace=False)
        loader = DataLoader(split_dataset, batch_size=1, sampler=SubsetSequentialSampler(ids), **kwargs)

    return loader


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)

      
def print_network(net):
    num_params = 0
    num_params_train = 0

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)