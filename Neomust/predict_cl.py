import os
import re
import torch
import mhcnames
import argparse
import numpy as np
import pandas as pd
from torch import nn
from multiprocessing import Pool, cpu_count
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description='Prediction using NeoMUST.')
parser.add_argument('--input', type=str, required=True,
                    help='The input file to be predicted(Required columns: "hla", "peptide") (*.csv)')
parser.add_argument('--blosum62', type=str, required=True, help='The BLOSUM62 file (*.csv)')
parser.add_argument('--mhc_aa', type=str, required=True,
                    help='The MHC_pseudo-sequences file(Required columns: "allele", "sequence" ) (*.csv)')
parser.add_argument('--neomust_model', type=str, required=True, help='The trained NeoMUST model file (*.pt)')
parser.add_argument('--rank_database', type=str, required=True, help='The Rank database path (/path)')
parser.add_argument('--output', type=str, required=True, help='The output file (*.csv)')

parser.add_argument('--batch_size', default=2048, type=int, help='batch_size in pytorch Dataloader')
parser.add_argument('--num_workers', default=0, type=int, help='num_workers in pytorch Dataloader')
parser.add_argument('--pin_memory', default=False, type=bool, help='pin_memory in pytorch Dataloader')
parser.add_argument('--max_len', default=50000, type=int, help='Maximum length per task to be split')
parser.add_argument('--max_task', default=cpu_count(), type=int, help='Maximum number of parallel tasks')
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

        # gate1
        selected1 = self.dnn1(pmhc)
        gate_expert_output1 = torch.cat((experts_task1_o, experts_shared_o), dim=0)
        gate1_out = torch.einsum('abc, ba -> bc', gate_expert_output1, selected1)
        final_output1 = self.tower1(gate1_out)

        # gate2
        selected2 = self.dnn2(pmhc)
        gate_expert_output2 = torch.cat((experts_task2_o, experts_shared_o), dim=0)
        gate2_out = torch.einsum('abc, ba -> bc', gate_expert_output2, selected2)
        final_output2 = self.tower2(gate2_out)

        return final_output1, final_output2


def rank(rank_database_path, pseudo, df):
    rank_df = pd.read_csv(os.path.join(rank_database_path, pseudo + '.csv.gz'))
    df_ = pd.DataFrame(index=df.index)
    rank_el_list = []
    for i in df['cgc_pre_task2']:
        rank_el_list.append(rank_df.at[rank_df.index[(rank_df['el'] - i).abs().argmin()], 'rank_el'])
    df_['rank_el'] = rank_el_list
    return df_


def predict(test_file, blosum62_file, mhc_aa_file, neomust_model_file, rank_database_path, output_file):
    test_df = pd.read_csv(test_file)
    blosums, aa_dict = read_blosum_aa(blosum62_file)
    pad_zeros = get_paddings(blosums)
    mhc_aa_df = pd.read_csv(mhc_aa_file)
    mhc_dict = {row['allele']: row['sequence'] for index, row in mhc_aa_df.iterrows()}
    test_df['pseudo'] = [mhc_dict[mhcnames.normalize_allele_name(i)] for i in test_df['hla']]
    mhc_array = seq_array_concat(test_df['pseudo'].tolist(), aa_dict, blosums, pad_zeros, is_allele=True)
    pep_array = seq_array_concat(test_df['peptide'].tolist(), aa_dict, blosums, pad_zeros)
    ds_test = TensorDataset(torch.tensor(mhc_array), torch.tensor(pep_array))
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    model = CGC(input_size=256, num_specific_experts=4, num_shared_experts=4, experts_out=64, experts_hidden=128,
                towers_hidden=32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    if torch.cuda.device_count() > 1:
        model = model.cuda()
    model.load_state_dict(torch.load(neomust_model_file, map_location=device), strict=False)
    model.eval()
    cgc_pre_task1_list = []
    cgc_pre_task2_list = []
    for (features1, features2) in dl_test:
        with torch.no_grad():
            features1 = features1.to(device)
            features2 = features2.to(device)
            predictions = model(features1, features2)
            predictions1 = predictions[0]
            predictions2 = predictions[1]
        cgc_pre_task1_list.extend(to_ic50(predictions1.cpu().view(-1)).tolist())
        cgc_pre_task2_list.extend(predictions2.cpu().view(-1).tolist())
    test_df['neomust_ba'] = cgc_pre_task1_list
    test_df['neomust_el'] = cgc_pre_task2_list

    max_len = args.max_len
    pseudo_list = []
    df_list = []
    for pseudo, df in test_df.groupby('pseudo'):
        for i in range(0, df.shape[0], max_len):
            pseudo_list.append(pseudo)
            df_list.append(df.iloc[i:i + max_len, :])
    t = args.max_task
    pool = Pool(t)
    result_list = []
    for pseudo, df in zip(pseudo_list, df_list):
        result_list.append(pool.apply_async(rank, (rank_database_path, pseudo, df)))
    pool.close()
    pool.join()
    df_concat = pd.concat([i.get() for i in result_list])
    df_concat.sort_index(inplace=True)
    test_df['neomust_el_rank'] = df_concat['rank_el']
    del test_df['pseudo']
    test_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    predict(args.input, args.blosum62, args.mhc_aa, args.neomust_model, args.rank_database, args.output)
