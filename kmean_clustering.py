import torch
from torchaudio.transforms import MFCC
from kmeans_pytorch import kmeans
import glob
from tqdm.auto import tqdm
import torchaudio
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sample-rate', default=8000)
parser.add_argument('--n-fft', default=256)
parser.add_argument('--n-mfcc', default=25)
parser.add_argument('--n-mels', default=80)
parser.add_argument('--n-clusters', default=50)
parser.add_argument('--name', default='train_cd92')
parser.add_argument('--save-dir', default='/home/stud_vantuan/share_with_150/data_CD/cluster')
parser.add_argument('--data-dir', default='/home/stud_vantuan/share_with_150/data_CD/train_CD92/wav')
parser.add_argument('--gpu-id', default=0)
args = parser.parse_args()

SR = args.sample_rate
N_FFT = args.n_fft
HOP_LEN = N_FFT // 4
N_MFCC = args.n_mfcc
N_MELS = args.n_mels
NUM_CLUSTERS = args.n_clusters
NAME = args.name
SAVE_DIR = args.save_dir
device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

mfcc_tool = MFCC(
    sample_rate=SR,
    n_mfcc=N_MFCC,
    melkwargs={
        "n_fft": N_FFT,
        "n_mels": N_MELS,
        "hop_length": HOP_LEN,
        "mel_scale": "htk",
    }
)

def cal_mfcc(wav):
    return mfcc_tool(wav)

def do_kmeans(X, num_clusters):
    return kmeans(X=X, num_clusters=num_clusters,
                  distance='euclidean', device=device)

cd92_5h = glob.glob(f'{args.data_dir}/*.wav')
len(cd92_5h)

wavs = []
for i in tqdm(cd92_5h, desc='Loading wavs...'):
    w, sr = torchaudio.load(i)
    wavs.append(w)

mfcc_feats = []
for i in tqdm(wavs, desc='Calculate MFCCs...'):
    mfcc_feats.append(cal_mfcc(i)[0].mean(axis=-1))

X = torch.stack(mfcc_feats)

cluster_labels, cluster_centroids = do_kmeans(X, NUM_CLUSTERS)

save_path = f'KMeans_{NAME}_{NUM_CLUSTERS}.clusters'
with open(os.path.join(SAVE_DIR, save_path), 'w') as f:
    flag = False
    for p, l in zip(cd92_5h, cluster_labels.tolist()):
        if flag:
            f.write('\n')
        f.write(f'{p} {l}')
        flag = True
print("Write file... done!")