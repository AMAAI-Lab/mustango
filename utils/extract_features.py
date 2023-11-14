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
from utils.utils import BeatProcessor, ChordProcessor, get_bpm, get_key, format_key
from utils.prompt_functions import beats_prompt_fc, chords_prompt_fc, bpm_prompt_fc, key_prompt_fc
import json
from utils.keyfinder import Tonal_Fragment

import pyrubberband as pyrb

config_path = "/666/TANGO/gh/text2music/config/music_feat_extrator_config.yaml"

with open (config_path, 'r') as f:
    cfg = yaml.safe_load(f)



#load json, follow one by one
#extract chords
#extract beats
#save them (as an array, or as a new triplet json)
#add a new prompt for time signature and chord progression

# output_path='/666/TANGO/music-caps/data_enhanced'
data_path='/666/TANGO/music-caps/data'
ref_file='/666/TANGO/music-caps/test_musiccaps.json'
# os.makedirs(output_path,exist_ok=True)

out_json_path='/666/TANGO/music-caps/test_musiccaps_ep.json'
# json_continue_source='/666/TANGO/music-caps/train_musiccaps_ep.json'

beat_estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=["beat_particles"], thread=False)
beat_processor = BeatProcessor(beat_estimator, **cfg['beat_processor'])

chord_estimator = Chordino()  
chord_processor = ChordProcessor(chord_estimator,**cfg['chord_processor'])

chord_feats=[]
beat_feats=[]

continue_flag=False
continue_i=5072

with open(ref_file, encoding='utf-8') as ref:
    i=0
    j=0
    with open(out_json_path, 'w', encoding='utf-8') as out_json_file:
        new_row={}
        for row in ref:

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
            file_path=a['location']
            # print(row)
            cf=chord_processor(file_path)
            bf=beat_processor(file_path)
            bpm=get_bpm(bf)
            key,corr,altkey,altcorr=get_key(file_path)
            if corr<cfg['key_extraction']['correlation_threshold']:
                key=None
            # key,altkey=format_key([key,altkey])
            music_prompts = chords_prompt_fc(cf) + ' ' + beats_prompt_fc(bf) + ' ' + bpm_prompt_fc(bpm) + ' ' + key_prompt_fc(format_key([key,altkey]))
            music_prompts = music_prompts.replace('  ',' ', 3)
            new_row['dataset']='musiccaps_ep'
            # new_row['location']=os.path.join(output_path,os.path.splitext(os.path.basename(file_path))[0]+'_ep.wav')
            new_row['location']=a['location']
            new_row['old_captions']=a['captions']
            new_row['captions']=a['captions'] + ' ' + music_prompts
            new_row['music_feat_captions']=music_prompts
            new_row['aug_captions']=''
            new_row['beats']=[bf[0].tolist(),bf[1].tolist()]
            new_row['chords']=cf
            new_row['bpm']=bpm
            new_row['key']=[key,altkey]
            out_json_file.write(json.dumps(new_row) + '\n')

            print(i)
            i+=1





# paths = glob.glob(cfg['data_path']+"/*.wav")[:2]
