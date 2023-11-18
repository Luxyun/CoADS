import torch
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from sksurv.metrics import concordance_index_censored

sys.path.append('../')
from util.util import *
from util.eval import *
from util.metrics import *
from datasets.PathDataset import *
from tensorboardX import SummaryWriter


def get_options():
    parser = argparse.ArgumentParser(description='Configurations for WSI-wise survival Training')
    parser.add_argument("--model", type=str, default='CoADS', help='the model name')
    parser.add_argument("--mode", type=str, default='graph', choices=['path', 'graph', 'cluster', 'vit'])
    parser.add_argument("--backbone", type=str, default='50',help='the backbone name')
    
    parser.add_argument('--max_epochs', type=int, default=200, help='maximum number of epochs to train')
    parser.add_argument('--stop_epochs', type=int, default=50, help='minimum number of epochs to train')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--warmup', default=10, type=int, help='number of epochs to validate')
    parser.add_argument('--patience', default=30, type=int)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument("--reg", type=float, default=1e-5, help='the L1 regulation for model')
    parser.add_argument('--seed_reproduce', type=int, default=42, help='random seed for reproducible')
    parser.add_argument('--seed_data', type=int, default=0, help='the data split seed')
    parser.add_argument('--ks',type=int,default=0,help='the begin fold')
    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
    parser.add_argument('--bag_loss', type=str, choices=['nll_loss', 'cox_loss'], default='nll_loss')
    parser.add_argument('--opt', type=str, choices=['adam', 'sgd','adamW'], default='adam')
    parser.add_argument('--scheduler', type=str, choices=['step', 'cosine','min','warmcosine'], default='min')
    parser.add_argument('--alpha_surv', type=float, default=0, help='How much to upweight uncensored patients')
    parser.add_argument('--n_bins', type=int, default=4, help='number of bins to use for event-time discretization')
    parser.add_argument("--gpu", type=str, default='3', help='the gpu id')
    parser.add_argument("--pred_task", type=str, default='OS', help='the prediction task about prognosis')
    parser.add_argument('--data', default='LUAD', help='the datasets id')
    parser.add_argument('--mag', default=20, type=int,help='the magnification of WSI')
    parser.add_argument("--gc", type=int, default=32)
    parser.add_argument('--weighted_sample', type=bool, default=False, help='enable weighted sampling')
    parser.add_argument('--resample', type=float, default=0, help='randomly sample some patches from each WSI')
    parser.add_argument('--dropout', type=float, default=0.25,help='the drop rate of model')
    parser.add_argument('--gate', type=bool, default=True, help='whether to use gate')
    parser.add_argument("--log", type=bool, default=True, help='whether to log data')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--print_info', action='store_true', help='enable print info in each fold')
    args = parser.parse_args()
    return args,parser


def train_surv(datasets, cur, args):
    '''
    train for the single fold

    :param datasets: [train_dset/val_dset/test_dset]
    :param cur: [fold_number]
    :param args:
    :return: results_dict,test_c_index,valid_c_index
    '''
    reg_fn = None
    train_dset, val_dset, test_dset = datasets
    print(f'Training Fold {cur}!\tsamples={len(train_dset) + len(val_dset) + len(test_dset)}\t'
          f'train:valid:test = {len(train_dset)}:{len(val_dset)}:{len(test_dset)}')
    writer_dir = os.path.join(args.results_dir, str(cur))
    os.makedirs(writer_dir, exist_ok=True)

    if args.log:
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    model=get_model(args)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    # print_network(model)
    # print(model)
    
    train_loader = get_split_loader(train_dset, training=True,mode=args.mode,testing=args.debug)
    train_inf_loader = get_split_loader(train_dset, training=False, mode=args.mode,testing=args.debug)
    val_loader = get_split_loader(val_dset, training=False, mode=args.mode,testing=args.debug)
    test_loader = get_split_loader(test_dset, training=False, mode=args.mode,testing=args.debug)
    
    loss_fn = get_criterion(args.bag_loss,args.alpha_surv)
    optimizer = get_optim(model, args)
    scheduler = get_scheduler(optimizer,args)
    monitor_cindex = Monitor_CIndex(patience=args.patience, stop_epoch=args.stop_epochs, verbose=False)
    valid_loss = 0

    for epoch in range(args.max_epochs):
        train_loss, train_CI = train_survival(epoch, model, train_loader, optimizer, loss_fn, writer, args.gc, args.reg)
         
        if epoch >= args.warmup:
            valid_loss, valid_CI = validate_survival(epoch, model, val_loader, loss_fn, writer, args.reg, stage='val')
             
            monitor_cindex(epoch, valid_CI, model,ckpt_name=os.path.join(args.results_dir, "fold_{}.pt".format(cur)))
            if monitor_cindex.early_stop:
                break
             
            if args.print_info:
                if epoch != 0 and epoch % 10 == 0:
                    print(f'Epoch:{epoch}\tloss={train_loss:.4f}\ttrain_CI={train_CI:.4f}\tvalid_CI={valid_CI:.4f}\r')
            
            if args.scheduler == 'min':
                scheduler.step(valid_loss)   
            else:
                scheduler.step()
        

    model.load_state_dict(torch.load(os.path.join(args.results_dir, "fold_{}.pt".format(cur))))
    results_train_df, train_cindex, train_p = summary_survival(model, train_inf_loader, args.n_classes, stage='train')
    results_val_df, val_cindex, val_p= summary_survival(model, val_loader, args.n_classes, stage='valid')
    results_test_df, test_cindex, test_p = summary_survival(model, test_loader, args.n_classes, stage='test')

    results_df = [results_train_df, results_val_df, results_test_df]
    results_cindex = [train_cindex, val_cindex, test_cindex]
    results_p = [train_p, val_p, test_p]

    print(f'Fold:{cur}\ttrain_CI:{train_cindex:.4f}-{train_p:.4f}\tval_CI:{val_cindex:.4f}-{val_p:.4f}\ttest_CI:{test_cindex:.4f}-{test_p:.4f}')

    if writer:
        writer.add_scalar('cindex/train', train_cindex, cur)
        writer.add_scalar('cindex/test', test_cindex, cur)
        writer.add_scalar('cindex/val', val_cindex, cur)
        
    writer.close()
    return results_df, results_cindex, results_p

def train_survival(epoch, model, loader, optimizer, loss_fn=None, writer=None, gc=16, reg=0):
    model.train()
    train_loss, train_loss_surv = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_events = np.zeros((len(loader)))
    all_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, label, time, event) in enumerate(loader):
        data_WSI = data_WSI.cuda()
        label = label.cuda()
        event = event.cuda()
        hazards, S, Y_hat, _, _ = model(x_path=data_WSI)

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=event)
        loss_value = loss.item()

        loss_reg = l1_reg_all(model) * reg
        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg
        loss = loss / gc + loss_reg
        loss.backward()

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_events[batch_idx] = event.item()
        all_times[batch_idx] = time

        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    train_loss /= len(loader)
    train_loss_surv /= len(loader)
    c_index = concordance_index_censored((all_events).astype(bool), all_times, all_risk_scores, tied_tol=1e-08)[0]
  
    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)
        writer.add_scalar('lr/learning_rate', optimizer.param_groups[0]["lr"], global_step=epoch)

    return train_loss_surv, c_index

def validate_survival(epoch, model, loader, loss_fn, writer=None, reg=0., stage='val'):
    model.eval()
    val_loss, val_loss_surv = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_events = np.zeros((len(loader)))
    all_times = np.zeros((len(loader)))

    with torch.no_grad():
        for batch_idx, (data_WSI, label, time, event) in enumerate(loader):
            data_WSI = data_WSI.cuda()
            label = label.cuda()
            event = event.cuda()
            hazards, S, Y_hat, _, _ = model(x_path=data_WSI)
            
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=event, alpha=0)
            loss_value = loss.item()
            loss_reg = l1_reg_all(model) * reg
            val_loss_surv += loss_value
            val_loss += loss_value + loss_reg

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_events[batch_idx] = event.item()
            all_times[batch_idx] = time

        val_loss /= len(loader)
        val_loss_surv /= len(loader)
        c_index = concordance_index_censored((all_events).astype(bool), all_times, all_risk_scores,tied_tol=1e-08)[0]
        
        if writer:
            writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/c-index', c_index, epoch)

        return val_loss_surv, c_index


def summary_survival(model, loader, n_classes, stage='test'):
    model.eval()
    all_risks = np.zeros((len(loader)))
    all_events = np.zeros((len(loader)))
    all_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['case_id']
    patient_results = {}

    with torch.no_grad():
        for batch_idx, (data_WSI, label, time, event) in enumerate(loader):
            data_WSI = data_WSI.cuda()
            label = label.cuda()
            slide_id = slide_ids.iloc[batch_idx]
            hazards, S, Y_hat, _, _ = model(x_path=data_WSI)

            risk = -(torch.sum(S, dim=1).cpu().numpy()).item()
            time = time.item()
            event = event.item()
            label = label.item()
            all_risks[batch_idx] = risk
            all_events[batch_idx] = event
            all_times[batch_idx] = time
            patient_results.update({batch_idx: {'case_id': np.array(slide_id),'discrete_label': label,
                                                'risk': risk,'survival_time': time, 'event': event, 'stage': stage}
                                    })

        c_index = get_cindex(all_risks, all_times, all_events)
        p_value = cox_log_rank(all_risks, all_times, all_events)
        results_df = pd.DataFrame.from_dict(patient_results, orient='index')
    return results_df, c_index, p_value


def main(args):
    assert args.k == 5, print('The number of folds should be 5, this script is for the 5-fold cross validation')
    names, cindex, p_value,auc, HR = [], [], [], [],[]
    
    Fold_start=int(args.ks)
    Fold_total=int(args.k)
    if Fold_total==5 and Fold_start==0:
        assert args.k == 5, print('The number of folds should be 5, this script is for the 5-fold cross validation')
        for i in np.arange(args.ks,args.k):
            start = timer()
            split_csv_dir = os.path.join(args.split_csv_root, 'split_{}.csv'.format(i))
            datasets = MIL_Survival_Dataset(data_path=args.pth_dir, csv_path=args.csv_dir, pred_task=args.pred_task,
                                            mode=args.mode, n_bins=args.n_bins)
            train_dset, val_dset, test_dset = datasets.return_splits(from_id=False, csv_path=split_csv_dir)
            results_df, results_c, results_p = train_surv((train_dset, val_dset, test_dset), i, args)
            total_result = pd.concat([results_df[0], results_df[1], results_df[2]])
            total_result.to_csv(os.path.join(args.results_dir, 'total_split_{}_results.csv'.format(i)), index=False)
            end = timer()
            
            with open(os.path.join(args.results_dir, 'result_{}.txt'.format(i)), 'a') as f:
                f.write('fold:{}\ntest_cindex:{:.4f}\ntest_p:{:.4f}\n'.format(i, results_c[2], results_p[2]))

            cindex, p_value = map(lambda x, y: x + [y], [cindex, p_value], [results_c, results_p])

        np_cindex, np_p_value= map(lambda x: np.array(x), [cindex, p_value])
        mean_c, mean_p = map(lambda x: get_mean_std(x[:, 0], x[:, 1], x[:, 2]),[np_cindex, np_p_value])

        print(f"{'=' * 10} Training finished, {args.k} Fold Mean results : {'=' * 10}")
        print(f'train_c={mean_c[0][0]}±{mean_c[1][0]}, val_c={mean_c[0][1]}±{mean_c[1][1]},test_c={mean_c[0][2]}±{mean_c[1][2]}')

        names = [i for i in range(args.ks,args.k)]
        names.extend(['mean', 'std'])
        cindex=np_cindex.tolist()
        cindex.extend([mean_c[0], mean_c[1]])
        p_value=np_p_value.tolist()
        p_value.extend([mean_p[0], mean_p[1]])

        final_df = pd.DataFrame({'names': names, 'cindex': cindex, 'p_value': p_value})
        final_df[['train_c', 'val_c', 'test_c']] = final_df['cindex'].apply(pd.Series)
        final_df[['train_p', 'val_p', 'test_p']] = final_df['p_value'].apply(pd.Series)
        save_name = 'summary.csv'
        final_df.to_csv(os.path.join(args.results_dir, save_name), index=False)
        
    else:
        print(f'the total number of folds is {args.k}, the start fold is {args.ks}')
        for i in np.arange(args.ks,args.k):
            start = timer()
            split_csv_dir = os.path.join(args.split_csv_root, 'split_{}.csv'.format(i))
            datasets = MIL_Survival_Dataset(data_path=args.pth_dir, csv_path=args.csv_dir, pred_task=args.pred_task,
                                            mode=args.mode, n_bins=args.n_bins)
            train_dset, val_dset, test_dset = datasets.return_splits(from_id=False, csv_path=split_csv_dir)
            results_df, results_c, results_p= train_surv((train_dset, val_dset, test_dset), i, args)
            total_result = pd.concat([results_df[0], results_df[1], results_df[2]])
            total_result.to_csv(os.path.join(args.results_dir, 'total_split_{}_results.csv'.format(i)), index=False)
            end = timer()
            
            with open(os.path.join(args.results_dir, 'result_{}.txt'.format(i)), 'a') as f:
                f.write('fold:{}\ntest_cindex:{:.4f}\ntest_p:{:.4f}\n'.format(i, results_c[2], results_p[2]))


if __name__ == "__main__":
    args,parser = get_options()
    args = get_data(args)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.n_classes = args.n_bins
    seed_torch(args.seed_reproduce)

    os.makedirs(args.results_dir, exist_ok=True)
    print(f'csv path  -->{args.csv_dir}')
    print(f'split csv path  -->{args.split_csv_root}')
    print(f'data path -->{args.pth_dir}')
    print(f'result path -->{args.results_dir}')

    print_options(args, parser)

    start = timer()
    results = main(args)
    end = timer()
    print('Script Time: %f s, %d min' % (end - start, (end - start) / 60))
    print(f'checkpoint has been saved in {args.results_dir}')