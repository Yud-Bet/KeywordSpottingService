import librosa
import os
import json

from numpy import sign

DATASET_PATH = "recordings"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    # data dictionary
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # Loop through all the sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            category = dirpath.split("\\")[-1]
            data["mappings"].append(category)

            print(f"Processing {category}")
            for f in filenames:
                filepath = os.path.join(dirpath, f)
                signal, sr = librosa.load(filepath)
                if (len(signal) >= SAMPLES_TO_CONSIDER):
                    signal = signal[:SAMPLES_TO_CONSIDER]
                else:
                    signal = librosa.util.fix_length(signal, SAMPLES_TO_CONSIDER)
                
                MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
                data["labels"].append(i-1)
                data["MFCCs"].append(MFCCs.T.tolist())
                data["files"].append(filepath)
                print(f"{filepath}: {i - 1} length = {len(signal)}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)