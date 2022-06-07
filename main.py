from cmath import inf
import random

import torch
import numpy as np
import os
import setproctitle
import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
setproctitle.setproctitle('DeciMo@liuchang')

from torch.utils import data
from sklearn.metrics import roc_auc_score, log_loss, roc_curve, auc
from time import time
from prettytable import PrettyTable
from utils.parser import parse_args
from utils.data_loader import load_data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from gcn import Recommender
from utils.helper import early_stopping
from textDataset import textDataset

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0

def train(model, optimizer, data_loader, criterion, device):
    
    model.train()
    total_loss = 0
    cnt=0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        with torch.autograd.set_detect_anomaly(True):
        # field_fac, target_fac, field_cf, target_cf = field_fac.to(device), target_fac.to(device), field_cf.to(device), target_cf.to(device)
            fields, target = fields.to(device), target.to(device)
            y = model(fields, False)
            loss = criterion(y, target.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        cnt += 1
    total_loss = total_loss/cnt
    return total_loss

def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for (fields, target)in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields, True)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
        return roc_auc_score(targets, predicts), log_loss(targets, predicts)


if __name__=='__main__':

    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global args, device
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu_id) if args.cuda else torch.device("cpu"))

    n_params, graph = load_data(args)
    # adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    train_dataset = textDataset(args.dataset, 'train')
    valid_dataset = textDataset(args.dataset, 'valid')
    test_dataset = textDataset(args.dataset, 'test')
    field_dims = train_dataset.field_dims
    train_data_loader = DataLoader(train_dataset, batch_size = args.train_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size = args.test_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size = args.test_size, num_workers=8)

    model = Recommender(n_params, args, graph, field_dims).to(device)

    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    loss_list = []
    print("Start Training...")
    for epoch in range(args.epoch):
        train_s_t = time()
        total_loss = train(model, optimizer, train_data_loader, criterion, device)
        loss_list.append(total_loss)
        train_e_t = time()
        if epoch % 3 == 2 or epoch == 1:
            test_s_t = time()
            auc, logloss = test(model, valid_data_loader, device)
            test_e_t = time()
            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "testing time", "loss", "AUC", 'LogLoss']
            train_res.add_row(
                [epoch, train_e_t - train_e_t, test_e_t - test_s_t, total_loss, auc, logloss]
            )
            print(train_res)
            cur_best_pre_0, stopping_step, should_stop = early_stopping(auc, cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=2)
            if should_stop:
                break
        else:
            print('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, total_loss))
        auc, logloss = test(model, test_data_loader, device)
        print(f'test auc in epoch {epoch + 1}: {auc}')
        print(f'test logloss in epoch {epoch + 1}: {logloss}')
    torch.save(model.state_dict(), './param.pth')