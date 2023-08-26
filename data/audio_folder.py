"""
Implementing functions to collect valid audio files for creating dataset.
"""

import os
import os.path
from tqdm.auto import tqdm

AUDIO_EXTENSIONS = [
    '.wav', '.flac'
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def make_dataset_audio(dir, max_dataset_size=float("inf"), label_path=None):
    audios = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    if label_path == None:
        print("Walking on disk...")
        for root, _, fnames in tqdm(sorted(os.walk(dir)), desc='Fetching wavs...'):
            for fname in fnames:
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    audios.append((path, None))
        
        return audios[:min(max_dataset_size, len(audios))]
    
    print("Reading on a label...")
    with open(label_path, 'r') as f:
        for l in tqdm(f, desc="Fetching wavs..."):
            p, label = l.split(' ')
            audios.append((p, label))
    
    return audios