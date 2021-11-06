import os
import scipy.io.wavfile
import subprocess

PATH = 'C:/Users/yudbet/Desktop/one'

def change_file_name(dir_path, keyword, people):
    i = 0
    for filename in os.listdir(dir_path):
        os.rename(os.path.join(dir_path, filename), os.path.join(dir_path, f"{keyword}_{people}_{i}.m4a"))
        i += 1

def convert_to_wav8k1c(src_path, dst_path):
    for filename in os.listdir(src_path):
        name = filename.split('.')[0]
        command = f'ffmpeg/bin/ffmpeg -i {os.path.join(src_path, filename)} -ar 8000 -ac 1 {os.path.join(dst_path, name)}.wav'
        subprocess.call(command)

def convert_to_wav8k1c(file_path):
    dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    file_name = file_name.split('.')[0]
    command = f'ffmpeg/bin/ffmpeg -i {file_path} -ar 8000 -ac 1 {os.path.join(os.path.curdir, file_name)}.wav'
    subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def trim_silence(audio, noise_threshold=150):
    start = None
    end = None

    for idx, point in enumerate(audio):
        if abs(point) > noise_threshold:
            start = idx
            break

    # Reverse the array for trimming the end
    for idx, point in enumerate(audio[::-1]):
        if abs(point) > noise_threshold:
            end = len(audio) - idx
            break

    return audio[start:end]

def trim_silence_file(file_path, noise_threshold=150):
    rate, audio = scipy.io.wavfile.read(file_path)
    trimmed_audio = trim_silence(audio, noise_threshold=noise_threshold)
    scipy.io.wavfile.write(file_path, rate, trimmed_audio)

def trim_all_dir(dir_path):
    for filename in os.listdir(dir_path):
        trim_silence_file(os.path.join(dir_path, filename))

def preprocess(src_dir, dst_dir, keyword, people):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    change_file_name(src_dir, keyword, people)

    convert_to_wav8k1c(src_dir, dst_dir)

    trim_all_dir(dst_dir)

def prepare_data(file_path):
    convert_to_wav8k1c(file_path)
    trim_silence_file(os.path.join(os.path.curdir, os.path.basename(file_path).split('.')[0]) + '.wav')