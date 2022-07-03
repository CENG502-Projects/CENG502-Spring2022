import glob
import os
import librosa # Audio processing library
from tqdm import tqdm # creating Progress Meters or Progress Bars
import random
import numpy as np


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

  # We have 6 emotions rather than 8
  # In the paper, there is only 4 emotions, where they can combine happy and excitement.
LABEL_DICT1 = {
    '01': 'neutral',
    '02': 'frustration',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    # '06': 'fearful',
    '07': 'excitement'
    # '08': 'surprised'

impro_or_script='impro'
RATE=16000
T=2

def build_test_list(valid_files, LABEL_DICT1, RATE, t):
    testList=[]
    for i, wav_file in enumerate(tqdm(valid_files)):
        label = str(os.path.basename(wav_file).split('-')[2])
        if (label not in LABEL_DICT1):
            continue
        if (impro_or_script != 'all' and (impro_or_script not in wav_file)):
            continue
        wav_data, _ = librosa.load(wav_file, sr=RATE)
        if (t * RATE >= len(wav_data)):
            continue
        testList.append(os.path.basename(wav_file))
    return testList

# Load process wav and txt files from our local drive

def process_IEMO():
    wavs = glob.glob('C:\\Users\\hjjh\\PycharmProjects\\pythonProject\\CENG502-PROJECT\\iemocap\\*.wav')
    transes = glob.glob('C:\\Users\\hjjh\\PycharmProjects\\pythonProject\\CENG502-PROJECT\\iemocap*.txt')
    write_list = []
    for wav in tqdm(wavs):
        wav_name = os.path.basename(wav)
        wav_name_split = wav_name.split('.')[0].split('-')
        if(wav_name_split[2] not in LABEL_DICT1): # If the wav_name not found our label dictionary split the wav_name txt file.
            continue
        if ('script' in wav_name):
            txt_name = wav_name_split[0] + '_' + wav_name_split[1] + '_' + wav_name_split[-1].split('_')[0] + '.txt'
        else:
            txt_name = wav_name_split[0] + '_' + wav_name_split[1] + '.txt'
        trans_name = None
        for trans in transes:
            if (os.path.basename(trans) == txt_name):
                trans_name = trans
                break
        if (trans_name is not None):
            f_trans = open(trans_name)
            fr_trans = f_trans.readlines()
            FIND = False
            for l_trans in fr_trans:
                if (l_trans.split(' ')[0] == wav_name_split[0] + '_' + wav_name_split[1] + '_' + wav_name_split[-1]):
                    write_list.append((l_trans.split(' ')[0], l_trans.split(':')[-1].replace('\n',''), wav_name, wav_name_split[2]))
                    FIND = True
                    break
            f_trans.close()
    with open('IEMOCAP.csv', 'w') as f:
        for wl in write_list:
            for w in range(len(wl)):
                f.write(wl[w])
                if (w < len(wl) - 1):
                    f.write('\t')
                else:
                    f.write('\n')
