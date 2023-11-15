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
from utils import BeatProcessor, ChordProcessor, get_bpm, get_key, format_key
from prompt_functions import beats_prompt_fc, chords_prompt_fc, bpm_prompt_fc, key_prompt_fc
import json
from keyfinder import Tonal_Fragment

from essentia.standard import MonoLoader, KeyExtractor

import pyrubberband as pyrb
from glob import glob
# config_path = "/666/TANGO/gh/text2music/config/music_feat_extrator_config.yaml"
config_path = "/666/TANGO/tango/configs/music_feat_extrator_config.yaml"

with open (config_path, 'r') as f:
    cfg = yaml.safe_load(f)



#load json, follow one by one
#extract chords
#extract beats
#save them (as an array, or as a new triplet json)
#add a new prompt for time signature and chord progression

# output_path='/666/TANGO/music-caps/data_enhanced'
# data_path='/666/TANGO/gh/text2music/outputs/CTRLpreds_1696678534_trinitydrops-MC3aug_epoch_72_steps_200_guidance_3'
# data_path='/666/TANGO/gh/text2music/outputs/FMATangoMCHFB_1699707309_666_TANGO_gh_text2music_saved_TangoMusiccapsHF_steps_200_guidance_3'
data_path='/666/TANGO/gh/text2music/outputs/ELT30TMP_1699506735_666_TANGO_gh_text2music_saved_tangomusic_pretrained_epoch_30_steps_200_guidance_3'

# ref_file='/666/TANGO/gh/text2music/data/MC3LH_testB_preds.json'
ref_file='/666/TANGO/gh/text2music/data/expert_listening_test_predictions.json'
# os.makedirs(output_path,exist_ok=True)

out_json_path=ref_file.split('.')[0]+"TMP30.json"
# out_json_path='/666/TANGO/music-caps/output_ctrl3_feats.json'
# json_continue_source='/666/TANGO/music-caps/train_musiccaps_ep.json'

beat_estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=["beat_particles"], thread=False)
beat_processor = BeatProcessor(beat_estimator, **cfg['beat_processor'])

chord_estimator = Chordino()  
chord_processor = ChordProcessor(chord_estimator,**cfg['chord_processor'])

chord_feats=[]
beat_feats=[]

continue_flag=False
continue_i=5072
files=glob(data_path+"/*.wav")

new_list=[]
for entry in files:
    new_list.append(entry.split('_')[-1].split('.')[0])
indeces=np.argsort(np.array(new_list).astype(int))

# print(indeces)

with open(ref_file, encoding='utf-8') as ref:
    i=0
    j=0
    with open(out_json_path, 'w', encoding='utf-8') as out_json_file:
        new_row={}
        for row,index in zip(ref,indeces):

            if continue_flag:
                with open(json_continue_source, 'r', encoding='utf-8') as json_source:
                    for row2 in json_source:
                        if j==continue_i:
                            break
                        b=json.loads(row2)
                        out_json_file.write(json.dumps(b) + '\n')
                        j+=1
                    continue_flag=False
            if i<j:
                i+=1
                continue
            a=json.loads(row)
            # shutil.copy(a['location'],os.path.join(output_path,'output_{}.wav'.format(i)))
            file_path=files[index]
            # print(row)
            cf=chord_processor(file_path)
            bf=beat_processor(file_path)
            bpm=get_bpm(bf)
            key,corr,altkey,altcorr=get_key(file_path)
            if corr<cfg['key_extraction']['correlation_threshold']:
                key=None
            # key,altkey=format_key([key,altkey])
            # new_row['location']=os.path.join(output_path,os.path.splitext(os.path.basename(file_path))[0]+'_ep.wav')

            audio = MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()
            keyex=KeyExtractor(sampleRate=16000)
            detkey=keyex(audio)

            new_row=a.copy()
            new_row['name']=os.path.basename(files[index])
            # new_row['old_captions']=a['captions']
            # new_row['captions']=a['captions'] + ' ' + music_prompts
            # new_row['captions']=a['prompt']
            # new_row['feature']=a['feature']
            # new_row['beats_predicted']=a['beats_predicted']
            # new_row['chords_predicted']=a['chords_predicted']
            # new_row['chords_predicted_time']=a['chords_predicted_time']

            # new_row['beats']=[bf[0].tolist(),bf[1].tolist()]
            # new_row['chords']=cf
            # new_row['bpm']=bpm
            # new_row['key']=[key,altkey]

            new_row['beats_ext']=[bf[0].tolist(),bf[1].tolist()]
            # new_row['chords_ext']=cf
            new_row['bpm_ext']=bpm
            new_row['key_ext']=[key,altkey]
            # new_row['key2_ext']=[detkey]

            chords=cf
            c_type=[]
            c_time=[]
            for ch in chords:
                c_type.append(ch[0])
                c_time.append(ch[1])

            new_row['chords_ext']=c_type
            new_row['chords_time_ext']=c_time

            new_row['key2_ext']=[detkey[0],detkey[1]]
            new_row['key2prob_ext']=[detkey[2]]

            out_json_file.write(json.dumps(new_row) + '\n')

            print(i)
            i+=1





# paths = glob.glob(cfg['data_path']+"/*.wav")[:2]
