import nlpaug.augmenter.audio as naa
import librosa
import glob
from tqdm import tqdm
import os

# Source : https://nlpaug.readthedocs.io/en/latest/augmenter/audio/vtlp.html
# This part is not given explicitly in the paper.
# The default parameters are used.

wavlist=glob.glob('C:\\Users\\hjjh\\PycharmProjects\\pythonProject\\CENG502-PROJECT\\iemocap\/IEMOCAP/*.wav')
targetdirectory=  'C:\\Users\\hjjh\\PycharmProjects\\pythonProject\\CENG502-PROJECT\\iemocap\/IEMOCAP'
aug = naa.VtlpAug(40000, zone=(0.2, 0.8), coverage=0.1, fhi=4800, factor=(0.9, 1.1), name='Vtlp_Aug', verbose=0, stateless=True )
for w in tqdm(wavlist):
    for i in range(7):
        wav, _=librosa.load(w, 40000)
        wavAug=aug.augment(wav)
        wavName=os.path.basename(w)
        librosa.output.write_wav(targetdirectory+wavName+'.'+str(i+1), wavAug ,40000) # Sample frequency 40 kHz
