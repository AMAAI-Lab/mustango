from BeatNet.BeatNet import BeatNet
import soundfile as sf
import numpy as np 
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from chord_extractor.extractors import Chordino
import yaml
import os
import glob
from scipy.stats import cosine
import json

import pyrubberband as pyrb

from utils import pitch_change, volume_change, volume_change_simple, speed_change, chord_pitch_change_fc, chord_speed_change_fc, beat_change_fc, bpm_change_fc, key_change_fc, format_key, key_change_fc2
from prompt_functions import pitch_prompt_fc, speed_prompt_fc, cres_prompt_fc, beats_prompt_fc, chords_prompt_fc, bpm_prompt_fc, key_prompt_fc, key_prompt_fc2

config_path = "../configs/augmentation_config.yaml"
#config 
with open (config_path, 'r') as f:
    cfg = yaml.safe_load(f)

#PUT JSON HERE
# paths = glob.glob(cfg['data_path']+"/*.wav")[:2]

output_path='/music-caps/data_aug2'
data_path='/music-caps/data'
ref_file='/music-caps/MC3_train.json'
#read chords and beats from ep.json
#but read old prompts from orig.json
out_json_path='/music-caps/MC3_train_aug.json'

number_of_runs=cfg['augmentation']['number_of_runs']
do_all_pitch_shifts=cfg['full_pitch_shift']['do_all_shifts']
if do_all_pitch_shifts:
    assert 2*cfg['full_pitch_shift']['semitone_range']==cfg['augmentation']['number_of_runs'][2], "Semitone range and number of runs assigned to pitch shift should correspond if do_all_pitch_shifts is enabled."
    pitch_range=np.arange(-1*cfg['full_pitch_shift']['semitone_range'],cfg['full_pitch_shift']['semitone_range']+1)
os.makedirs(output_path,exist_ok=True)
with open(ref_file, encoding='utf-8') as ref:
    with open(out_json_path, 'w', encoding='utf-8') as out_json_file:
        new_row={}
        i=0
        for row in ref:
            a=json.loads(row)
            # shutil.copy(a['location'],os.path.join(output_path,'output_{}.wav'.format(i)))
            cur_fil=a['location']
            cur_file_name = os.path.basename(cur_fil)
            if do_all_pitch_shifts:
                daps=0

            for j in range(np.sum(number_of_runs)):
                new_row=a.copy()
                y, sr = sf.read(cur_fil)
                chords=a['chords'] #load the feats
                beats=(np.array(a['beats'][0]),np.array(a['beats'][1]))
                bpm=a['bpm']
                # key,keytype,probs=a['key2'][0]
                key,altkey=a['key']

                if cfg['augmentation']['random']:
                    which_aug=np.zeros(6)
                    if cfg['augmentation']['if_only_one_type']:
                        aug_ind=np.random.choice(6,p=cfg['augmentation']['aug_distribution'])
                        which_aug[aug_ind]=1
                    else:
                        which_aug=np.random.randint(0,2,6)
                        #how_many=np.random.randint(1,4)
                        #todo generalised

                else:
                    which_aug=np.zeros(4)
                    aug_ind=np.sum(j>(np.cumsum(number_of_runs)-1))
                    which_aug[aug_ind]=1
                aug_prompts=''
                if which_aug[0]==1:
                    #no augmentation, just copy
                    # print('no augmentation, copying input')
                    aug_prompts=''
                if which_aug[1]==1:
                    #speed - tempo change the whole track
                    samp_len=len(y)
                    speed_start=0
                    speed_end=samp_len/sr
                    # speed_shift=np.random.randint(cfg['full_speed_change']['low_range']*10,cfg['full_speed_change']['high_range']*10)/10 # acceleration
                    speed_shift=np.random.uniform(cfg['full_speed_change']['low_range'],cfg['full_speed_change']['high_range'])
                    if np.random.randint(0,2)==0:
                        speed_shift=1/speed_shift # revert to slow down instead
                    speed_triplets = [(speed_shift,speed_start,speed_end)] #rate, start, end; todo for multiple

                    y, new_speed_triplets = speed_change(y, sr, speed_triplets)
                    beats = beat_change_fc(beats,new_speed_triplets,cfg['augmentation']['crop'])
                    bpm = bpm_change_fc(bpm,new_speed_triplets)
                    chords = chord_speed_change_fc(chords,new_speed_triplets,cfg['augmentation']['crop'])


                elif which_aug[2]==1:
                    samp_len=len(y)
                    pitch_start=0
                    pitch_end=samp_len/sr
                    if do_all_pitch_shifts:
                        pitch_range
                        pitch_shift=pitch_range[daps]
                        if pitch_shift==0:
                            daps+=1
                            pitch_shift=pitch_range[daps]
                        daps+=1

                    else:
                        # random mode
                        #pitch - shift the whole track
                        pitch_shift=np.random.randint(1,cfg['full_pitch_shift']['semitone_range']+1)
                        if np.random.randint(0,2)==0:
                            pitch_shift=-1*pitch_shift


                    pitch_triplets = [(pitch_shift,pitch_start,pitch_end)] #rate, start, end; todo for multiple

                    y = pitch_change(y,sr,pitch_triplets)
                    chords = chord_pitch_change_fc(chords,pitch_triplets)
                    key,altkey = key_change_fc([key,altkey],pitch_triplets)
                    new_row['key2'] = key_change_fc2(a['key2'],pitch_triplets)


                    #predetermined mode
                    #do all possible shifts in a range


                # elif which_aug[3]==1:
                #     #speed - partial change
                #     #change beat feature with speed...
                #     samp_len=len(y)
                #     #sample from cosine
                #     speed_start=cosine.rvs(loc=np.int32((samp_len-1)/2),scale=samp_len/(2*np.pi),size=1)/sr #todo for multiple
                #     speed_end=samp_len/sr
                #     # speed_shift=np.random.randint(cfg['speed_change']['low_range']*10,cfg['speed_change']['high_range']*10)/10
                #     speed_shift=np.random.randint(cfg['speed_change']['low_range']*10,cfg['speed_change']['high_range']*10)/10 # acceleration
                #     if np.random.randint(0,2)==0:
                #         speed_shift=1/speed_shift # revert for decceleration
                #     speed_triplets = [(speed_shift,speed_start,speed_end)] #rate, start, end; todo for multiple

                #     y, new_speed_triplets = speed_change(y, sr, speed_triplets)
                #     aug_prompts = aug_prompts + ' ' + speed_prompt_fc(new_speed_triplets,sr,cfg['augmentation']['crop'])
                #     beats = beat_change_fc(beats,new_speed_triplets,cfg['augmentation']['crop'])
                #     chords = chord_speed_change_fc(chords,new_speed_triplets,cfg['augmentation']['crop'])
                #     bpm = None

                # elif which_aug[4]==1:
                #     #pitch - sudden transposition
                #     # only from 20 to 80%? 3/5
                #     #change chords feature with pitch...
                #     samp_len=len(y)
                #     #sample from cosine
                #     #make everything np.array?
                #     pitch_start=cosine.rvs(loc=np.int32((samp_len-1)/2),scale=3*samp_len/(5*2*np.pi),size=1)/sr #todo for multiple
                #     # pitch_start=cosine.rvs(loc=np.int32((samp_len-1)/2),scale=samp_len/(2*np.pi),size=1)/sr #todo for multiple
                #     pitch_end=samp_len/sr
                #     pitch_shift=np.random.randint(-cfg['pitch_shift']['semitone_range'],cfg['pitch_shift']['semitone_range']+1)
                #     pitch_triplets = [(pitch_shift,pitch_start,pitch_end)] #rate, start, end; todo for multiple

                #     y = pitch_change(y,sr,pitch_triplets)
                #     aug_prompts = aug_prompts + ' ' + pitch_prompt_fc(pitch_triplets,sr)
                #     chords = chord_pitch_change_fc(chords,pitch_triplets)


                elif which_aug[3]==1:
                    #volume
                    #either from the start, or to the end...
                    dice = np.random.randint(0,2)
                    samp_len=len(y)
                    #sample from cosine
                    if dice==0: #change till the end
                        cres_start=float(cosine.rvs(loc=np.int32((samp_len-1)/2),scale=samp_len/(2*np.pi),size=1)/sr) #todo for multiple
                        cres_end=samp_len/sr
                    else: #change from start - only cres?
                        cres_start=0
                        cres_end=float(cosine.rvs(loc=np.int32((samp_len-1)/2),scale=samp_len/(2*np.pi),size=1)/sr) #todo for multiple
                    # cres_shift=np.random.randint(-cfg['volume_change']['semitone_range'],cfg['volume_change']['semitone_range']+1)
                    # cres_triplets = (pitch_shift,pitch_start,pitch_end) #rate, start, end; todo for multiple
                    # y = volume_change(cres_triplets)
                    # aug_prompts.append(cres_prompt_fc(cres_triplets))

                    if np.random.randint(0,2)==0:
                        rate_start=np.random.randint(1,5)*0.1
                        rate_end=1
                    else:
                        rate_start=1
                        rate_end=np.random.randint(1,5)*0.1
                    cres_quadruple = [(rate_start,rate_end,cres_start,cres_end)] #rate, rate, start, end; todo for multiple
                    y = volume_change_simple(y,sr,cres_quadruple,expo=cfg['volume_change']['exponential'])
                    aug_prompts = aug_prompts + ' ' + cres_prompt_fc(cres_quadruple,sr)
                    # chords_new[j]=
                    #now, for each sample, loop through aug X times and create new entries for a new json
                    #in the end concat the orig json with the aug one


                music_prompts = chords_prompt_fc(chords) + ' ' + beats_prompt_fc(beats) + ' ' + bpm_prompt_fc(bpm) + ' ' + key_prompt_fc2(a['key2'])
                music_prompts = music_prompts.replace('  ',' ', 3)
                if len(aug_prompts)>1:
                    if aug_prompts[0]==' ':
                        aug_prompts=aug_prompts[1:]
                if len(music_prompts)>1:
                    if music_prompts[0]==' ':
                        music_prompts=music_prompts[1:]

                # new_row['dataset']='musiccaps_aug'
                # new_row['location']=os.path.join(output_path,os.path.splitext(os.path.basename(file_path))[0]+'_ep.wav')
                # new_row['location']=a['location'] #new folder location!
                new_row['location']=os.path.join(output_path,cur_file_name.split('.')[0]+"_{}.wav".format(j))
                # new_row['old_captions']=a['old_captions']
                # new_row['captions']=(a['old_captions'] + ' ' + music_prompts + ' ' + aug_prompts).replace('  ',' ',3)
                new_row['music_feat_captions']=music_prompts
                new_row['aug_captions']=aug_prompts.replace('  ',' ',3)
                new_row['beats']=[beats[0].tolist(),beats[1].tolist()]
                new_row['chords']=chords
                new_row['bpm']=bpm
                new_row['prompt_bt']=beats_prompt_fc(beats)
                new_row['prompt_bpm']=bpm_prompt_fc(bpm)
                new_row['prompt_key2']=key_prompt_fc2(new_row['key2'])
                new_row['prompt_ch']=chords_prompt_fc(chords)
                new_row['key']=[key,altkey]
                # new_row['prompt_key']=key_prompt_fc(format_key([key,altkey]))

                # out_json_file.write(json.dumps(new_row) + '\n')
                out_json_file.write(json.dumps(new_row) + '\n')

                #write audio
                if len(y)>cfg['augmentation']['crop']*sr:
                    y=y[0:cfg['augmentation']['crop']*sr]
                write(os.path.join(output_path,cur_file_name.split('.')[0]+"_{}.wav".format(j)), sr, y.astype(np.float32))
            if np.mod(i,50)==0:
                print(i)
            i+=1
