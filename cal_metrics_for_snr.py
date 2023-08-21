import json
import argparse
import os
from tqdm.auto import tqdm
from metrics.evaluate_lsd import main as lsd
from metrics.mssl import main as mssl

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gan-noisy-path",
    default="/data/tuanio/data/cache/librispeech/train_10h/clean/test",
)
parser.add_argument(
    "--manual-noisy-path",
    default="/data/tuanio/data/librispeech/train_10h_different_snr/test",
)
args = parser.parse_args()

list_snrs = [-6, -3, 0, 3, 6]

results = []

predict = args.gan_noisy_path

for snr_db in tqdm(list_snrs, desc="Looping through snr..."):
    ground_truth = os.path.join(args.manual_noisy_path, str(snr_db) + "dB")

    avg_lsd, std_lsd, list_lsd = lsd(ground_truth, predict, use_gender=False)
    avg_mssl, std_mssl, list_mssl = mssl(ground_truth, predict, use_gender=False)

    results.append(
        dict(
            snr_db=snr_db,
            ground_truth_path=ground_truth,
            predict_path=predict,
            list_lsd=list_lsd,
            list_mssl=list_mssl,
            mean_lsd_mssl=(np.array(list_lsd) + np.array(list_mssl) / 2).tolist(),
        )
    )

with open("analyst_result.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
