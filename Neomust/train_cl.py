import os
import re
import copy
import torch
import random
import datetime
import warnings
import mhcnames
import argparse
import numpy as np
import pandas as pd
from torch import nn
from scipy.optimize import minimize
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Training using NeoMUST.')
parser.add_argument('--input', type=str, required=True,
                    help='The input file to be trained(Required columns: "hla", "peptide", "af", "ms", "measurement_inequality", "measurement_kind") (*.csv)')
parser.add_argument('--blosum62', type=str, required=True, help='The BLOSUM62 file (*.csv)')
parser.add_argument('--mhc_aa', type=str, required=True,
                    help='The MHC_pseudo-sequences file(Required columns: "allele", "sequence" ) (*.csv)')
parser.add_argument('--output_path', type=str, required=True, help='NeoMUST model output absolute path')
parser.add_argument('--batch_size', default=1024, type=int, help='batch_size in pytorch Dataloader')
parser.add_argument('--pin_memory', default=False, type=bool, help='pin_memory in pytorch Dataloader')
parser.add_argument('--max_epochs', default=60, type=bool, help='Maximum epochs')

args = parser.parse_args()


def read_blosum_aa(blosum_file):
    with open(blosum_file, "r") as f:
        blosums = []
        aa_dict = dict()
        index = 0
        for line in f:
            blosum_list = []
            line = re.sub("\n", "", line)
            for info in re.split("\s+", line):
                try:
                    blosum_list.append(float(info))
                except Exception as e:
                    if info not in aa_dict and info.isalpha():
                        aa_dict[info] = index
                        index += 1
            if len(blosum_list) > 0:
                blosums.append(blosum_list)
    return blosums, aa_dict


def get_paddings(blosums):
    length = len(blosums[0])
    pad_zeros = np.zeros(length, dtype=np.float32)
    return pad_zeros


def seq_aa_mapping(aa_seq, aa_dict, blosums, pad_zeros):
    seq_array = []
    for aa in aa_seq:
        seq_array.append(blosums[aa_dict[aa]])
    if len(seq_array) != 37:
        seq_array_left = seq_array
        for i in range(15 - len(seq_array)):
            seq_array_left = np.vstack((seq_array_left, pad_zeros))
        seq_array_center = seq_array
        for i in range((15 - len(seq_array)) // 2):
            seq_array_center = np.vstack((pad_zeros, seq_array_center))
        for i in range((15 - len(seq_array)) - ((15 - len(seq_array)) // 2)):
            seq_array_center = np.vstack((seq_array_center, pad_zeros))
        seq_array_right = seq_array
        for i in range(15 - len(seq_array)):
            seq_array_right = np.vstack((pad_zeros, seq_array_right))
        return np.vstack((seq_array_left, seq_array_center, seq_array_right))
    else:
        return np.asarray(seq_array, dtype=np.float32)


def seq_array_concat(seq_list, aa_dict, blosums, pad_zeros, is_allele=False):
    if not is_allele:
        seq_len = 45
    else:
        seq_len = 37
    pos = 0
    all_seq_array = np.zeros((len(seq_list), seq_len, 21), dtype=np.float32)
    for aa_seq in seq_list:
        all_seq_array[pos:pos + 1] = seq_aa_mapping(aa_seq, aa_dict, blosums, pad_zeros)
        pos += 1
    return all_seq_array


def from_ic50(ic50_list, max_ic50=50000.0):
    x = 1.0 - (np.log(np.maximum(ic50_list, 1e-16)) / np.log(max_ic50))
    return np.minimum(1.0, np.maximum(0.0, x))


def to_ic50(x, max_ic50=50000.0):
    return max_ic50 ** (1.0 - x)


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.selu = nn.SELU()
        self.dropout = nn.Sequential(nn.BatchNorm1d(hidden_size), nn.Dropout(0.2))

    def forward(self, x):
        out = self.fc1(x)
        out = self.selu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.selu = nn.SELU()
        self.dropout = nn.Sequential(nn.BatchNorm1d(hidden_size), nn.Dropout(0.2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.selu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


class CGC(nn.Module):
    def __init__(self, input_size, num_specific_experts, num_shared_experts, experts_out, experts_hidden,
                 towers_hidden):
        super(CGC, self).__init__()
        self.mhc_lstm = nn.LSTM(input_size=21, hidden_size=32, num_layers=2, bidirectional=True, batch_first=True)
        self.pep_lstm = nn.LSTM(input_size=21, hidden_size=32, num_layers=2, bidirectional=True, batch_first=True)
        self.input_size = input_size
        self.dropout = nn.Sequential(nn.BatchNorm1d(input_size), nn.Dropout(0.2))
        self.experts_hidden = experts_hidden
        self.experts_out = experts_out
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.towers_hidden = towers_hidden
        self.experts_shared = nn.ModuleList(
            [Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_shared_experts)])
        self.experts_task1 = nn.ModuleList(
            [Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_specific_experts)])
        self.experts_task2 = nn.ModuleList(
            [Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_specific_experts)])
        self.dnn1 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                  nn.Softmax(dim=1))
        self.dnn2 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                  nn.Softmax(dim=1))
        self.tower1 = Tower(self.experts_out, 1, self.towers_hidden)
        self.tower2 = Tower(self.experts_out, 1, self.towers_hidden)

    def shared_modules(self):
        return [self.mhc_lstm, self.pep_lstm, self.experts_shared]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def forward(self, x, y):
        self.mhc_lstm.flatten_parameters()
        self.pep_lstm.flatten_parameters()
        mhc_out, (mhc_h_n, mhc_c_n) = self.mhc_lstm(x)
        pep_out, (pep_h_n, pep_c_n) = self.pep_lstm(y)

        mhc_map = torch.cat([mhc_h_n[i, :, :] for i in range(mhc_h_n.shape[0])], dim=-1)
        pep_map = torch.cat([pep_h_n[i, :, :] for i in range(pep_h_n.shape[0])], dim=-1)
        pmhc = torch.cat([mhc_map, pep_map], dim=1)

        pmhc = self.dropout(pmhc)

        experts_shared_o = [e(pmhc) for e in self.experts_shared]
        experts_shared_o = torch.stack(experts_shared_o)
        experts_task1_o = [e(pmhc) for e in self.experts_task1]
        experts_task1_o = torch.stack(experts_task1_o)
        experts_task2_o = [e(pmhc) for e in self.experts_task2]
        experts_task2_o = torch.stack(experts_task2_o)

        selected1 = self.dnn1(pmhc)
        gate_expert_output1 = torch.cat((experts_task1_o, experts_shared_o), dim=0)
        gate1_out = torch.einsum('abc, ba -> bc', gate_expert_output1, selected1)
        final_output1 = self.tower1(gate1_out)

        selected2 = self.dnn2(pmhc)
        gate_expert_output2 = torch.cat((experts_task2_o, experts_shared_o), dim=0)
        gate2_out = torch.einsum('abc, ba -> bc', gate_expert_output2, selected2)
        final_output2 = self.tower2(gate2_out)

        return final_output1, final_output2


class MSEWithInequalitiesLoss(nn.Module):
    def __init__(self):
        super(MSEWithInequalitiesLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, y_pred, y_true):
        y_pred.to(self.device)
        y_true.to(self.device)
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        diff1 = y_pred - y_true
        diff1 *= (y_true >= 0.0).type(torch.FloatTensor).to(self.device)
        diff1 *= (y_true <= 1.0).type(torch.FloatTensor).to(self.device)
        diff2 = y_pred - (y_true - 2.0)
        diff2 *= (y_true >= 2.0).type(torch.FloatTensor).to(self.device)
        diff2 *= (y_true <= 3.0).type(torch.FloatTensor).to(self.device)
        diff2 *= (diff2 < 0.0).type(torch.FloatTensor).to(self.device)
        diff3 = y_pred - (y_true - 4.0)
        diff3 *= (y_true >= 4.0).type(torch.FloatTensor).to(self.device)
        diff3 *= (diff3 > 0.0).type(torch.FloatTensor).to(self.device)
        denominator = torch.maximum(torch.sum(torch.not_equal(y_true, 2.0).type(torch.FloatTensor).to(self.device), 0),
                                    torch.tensor(1.0))
        return (torch.sum(torch.square(diff1)) + torch.sum(torch.square(diff2)) + torch.sum(
            torch.square(diff3))) / denominator


def build_masked_loss(loss_function, mask_value=-1):
    def masked_loss_function(y_pred, y_true):
        mask = torch.as_tensor(torch.not_equal(y_true, mask_value), dtype=torch.float32)
        return loss_function(y_pred * mask, y_true * mask)

    return masked_loss_function


def cagrad(grads, num_tasks, alpha=0.5, rescale=1):
    GG = grads.t().mm(grads).cpu()
    g0_norm = (GG.mean() + 1e-8).sqrt()

    x_start = np.ones(num_tasks) / num_tasks
    bnds = tuple((0, 1) for x in x_start)
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (x.reshape(1, num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + c * np.sqrt(
            x.reshape(1, num_tasks).dot(A).dot(x.reshape(num_tasks, 1)) + 1e-8)).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha ** 2)
    else:
        return g / (1 + alpha)


def grad2vec(m, grads, grad_dims, task):
    grads[:, task].fill_(0.0)
    cnt = 0
    for mm in m.module.shared_modules():
        for p in mm.parameters():
            grad = p.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1


def overwrite_grad(m, newgrad, grad_dims):
    newgrad = newgrad * 2
    cnt = 0
    for mm in m.module.shared_modules():
        for param in mm.parameters():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1


def train(blosum62_file, input_file, mhc_aa_file, output_path):
    blosums, aa_dict = read_blosum_aa(blosum62_file)
    pad_zeros = get_paddings(blosums)
    input_df = pd.read_csv(input_file)
    mhc_aa_df = pd.read_csv(mhc_aa_file)
    mhc_dict = {row['allele']: row['sequence'] for index, row in mhc_aa_df.iterrows()}
    input_df['pseudo'] = [mhc_dict[mhcnames.normalize_allele_name(i)] for i in input_df['hla']]
    train_df = input_df.sample(frac=0.9)
    valid_df = input_df[~input_df.index.isin(train_df.index)]
    
    train_mhc_array = seq_array_concat(train_df['pseudo'].tolist(), aa_dict, blosums, pad_zeros, is_allele=True)
    train_pep_array = seq_array_concat(train_df['peptide'].tolist(), aa_dict, blosums, pad_zeros)
    valid_mhc_array = seq_array_concat(valid_df['pseudo'].tolist(), aa_dict, blosums, pad_zeros, is_allele=True)
    valid_pep_array = seq_array_concat(valid_df['peptide'].tolist(), aa_dict, blosums, pad_zeros)
    train_ic50_list = np.array(train_df['af'], dtype=np.float32)
    train_af_array = from_ic50(ic50_list=train_ic50_list)
    train_af_array = np.asarray([i[0] if i[0] == -1 else i[1] for i in zip(train_ic50_list, train_af_array)],
                                dtype=np.float32)
    valid_ic50_list = np.array(valid_df['af'], dtype=np.float32)
    valid_af_array = from_ic50(ic50_list=valid_ic50_list)
    valid_af_array = np.asarray([i[0] if i[0] == -1 else i[1] for i in zip(valid_ic50_list, valid_af_array)],
                                dtype=np.float32)
    train_ms_array = np.array(train_df['ms'], dtype=np.float32)
    valid_ms_array = np.array(valid_df['ms'], dtype=np.float32)
    train_offsets_array = train_df['measurement_inequality'].map({'=': 0, '<': 2, '>': 4, 0: 0}).values
    valid_offsets_array = valid_df['measurement_inequality'].map({'=': 0, '<': 2, '>': 4, 0: 0}).values
    
    ds_train = TensorDataset(torch.tensor(train_mhc_array),
                             torch.tensor(train_pep_array),
                             torch.tensor(train_af_array),
                             torch.tensor(train_ms_array),
                             torch.tensor(train_offsets_array))
    ds_valid = TensorDataset(torch.tensor(valid_mhc_array),
                             torch.tensor(valid_pep_array),
                             torch.tensor(valid_af_array),
                             torch.tensor(valid_ms_array),
                             torch.tensor(valid_offsets_array))
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, num_workers=0, pin_memory=args.pin_memory)
    dl_valid = DataLoader(ds_valid, batch_size=args.batch_size, num_workers=0, pin_memory=args.pin_memory)
    
    model = CGC(input_size=256, num_specific_experts=4, num_shared_experts=4, experts_out=64, experts_hidden=128,
                towers_hidden=32)
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
    
    loss_func1 = build_masked_loss(MSEWithInequalitiesLoss())
    loss_func2 = build_masked_loss(nn.BCELoss())
    coef_var = torch.tensor([1.0] * 2, requires_grad=True, device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(list(model.parameters()) + [coef_var], lr=1e-3, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, min_lr=1e-6, verbose=True)
    epochs = args.max_epochs
    grad_dims = []
    for mm in model.module.shared_modules():
        for param in mm.parameters():
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), 2).cuda()
    dfhistory = pd.DataFrame(
        columns=["epoch", 'loss_af', 'loss_ms', 'val_loss_af', 'val_loss_ms', 'coef_var_af', 'coef_var_ms'])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========" * 8 + "%s" % nowtime)
    for epoch in range(1, epochs + 1):
        model.train()
        loss1_sum = 0.0
        loss2_sum = 0.0
        step = 1
        for step, (features1, features2, labels1, labels2, offsets) in enumerate(dl_train, 1):
            features1 = features1.to(device)
            features2 = features2.to(device)
            labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            offsets = offsets.to(device)
            optimizer.zero_grad()
            predictions = model(features1, features2)
            predictions1 = predictions[0].view(-1)
            predictions2 = predictions[1].view(-1)
            loss1 = loss_func1(predictions1, labels1 + offsets)
            loss2 = loss_func2(predictions2, labels2)
            loss_tmp = [0, 0]
            loss_tmp[0] = loss1 / (2 * coef_var[0] ** 2) + torch.log(1 + coef_var[0] ** 2)
            loss_tmp[1] = loss2 / (coef_var[1] ** 2) + torch.log(1 + coef_var[1] ** 2)
            for i in range(2):
                if i == 0:
                    loss_tmp[i].backward(retain_graph=True)
                else:
                    loss_tmp[i].backward()
                grad2vec(model, grads, grad_dims, i)
                model.module.zero_grad_shared_modules()
            g = cagrad(grads, 2, alpha=0.8, rescale=1)
            overwrite_grad(model, g, grad_dims)
            optimizer.step()
            loss1_sum += loss1.item()
            loss2_sum += loss2.item()
        model.eval()
        val_loss1_sum = 0.0
        val_loss2_sum = 0.0
        val_step = 1
        for val_step, (features1, features2, labels1, labels2, offsets) in enumerate(dl_valid, 1):
            with torch.no_grad():
                features1 = features1.to(device)
                features2 = features2.to(device)
                labels1 = labels1.to(device)
                labels2 = labels2.to(device)
                offsets = offsets.to(device)
                predictions = model(features1, features2)
                predictions1 = predictions[0].view(-1)
                predictions2 = predictions[1].view(-1)
            val_loss1 = loss_func1(predictions1, labels1 + offsets)
            val_loss2 = loss_func2(predictions2, labels2)
            val_loss1_sum += val_loss1.item()
            val_loss2_sum += val_loss2.item()
        scheduler.step(val_loss2_sum / val_step)
        info = (
            epoch, loss1_sum / step, loss2_sum / step, val_loss1_sum / val_step, val_loss2_sum / val_step,
            coef_var[0].item(), coef_var[1].item())
        dfhistory.loc[epoch - 1] = info
        print((
                  "\nEPOCH = %d, loss_af = %.3f, loss_ms = %.3f, val_loss_af = %.3f, val_loss_ms = %.3f, coef_var_af = %.3f, coef_var_ms = %.3f") % info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % nowtime)
        print('Finished Training...')
        torch.save(model.state_dict(), f'{output_path}/epoch{epoch}.pt')
        dfhistory.to_csv(f'{output_path}/history.csv', index=False)

if __name__ == '__main__':
    train(args.blosum62, args.input, args.mhc_aa, args.output_path)
