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
# from utils.utils import BeatProcessor, ChordProcessor, get_bpm, get_key, format_key
from utils.prompt_functions import beats_prompt_fc, chords_prompt_fc, bpm_prompt_fc, key_prompt_fc
import json
# from utils.keyfinder import Tonal_Fragment

import pyrubberband as pyrb

# prompt_json_path='/666/TANGO/gh/text2music/data/MC3LH_testB_preds30epoTMP.json'
prompt_json_path='/666/TANGO/gh/text2music/data/fma_captioned_B_predictionsTMCHF.json'

out_json_path=prompt_json_path

def is_subsequence(A, B):
    m, n = len(A), len(B)
    i = j = 0
    
    while i < m and j < n:
        if A[i] == B[j]:
            j += 1
        i += 1
    
    return j


#COMPARE TWO JSONS - the one from prompt generator, and the one from feature extraction on generated audio
tempo_list = ['Grave', 'Largo', 'Adagio', 'Andante', 'Moderato', 'Allegro', 'Vivace', 'Presto', 'Prestissimo']
tempo_marks=np.array((40, 60, 70, 90, 110, 140, 160, 210))

maj_type_list = ['maj',' maj','M','major',' major']
min_type_list = ['m',' min','min','minor',' minor']

note_dup_list={"A": 0, "A#": 1, "Bb":1, "B":2, "Cb": 2, "B#":3, "C":3, "C#":4, "Db":4, "D": 5, "D#":6, "Eb":6, "E": 7 , "Fb": 7, "E#": 8, "F":8, "F#":9, "Gb":9, "G":10, "G#":11, "Ab":11, "N":99}

chord_type_dict={'m7b5/':2, 'major':1, 'm':2, '7':1, '6':1, 'dim':3, None:99, 'N':99, 'aug':3, 'maj7':1, '7/':1, 'm7b5':2, 'm6':2, 'm7':2, '/':1}

tempo_bin=[]
tempo_bin_oneoff=[]
bpm_deviation_abs=[]
bpm_deviation_rel=[]
key_counter=[]
key_counter_dup=[]
key_root_counter=[]
key_type_counter=[]
key_none_list=[]
chords_exact_counter=[]
chords_root_counter=[]
chord_gt_numbers=[]
chord_deviation_numbers=[]
chord_exact_match_counter=[]
chord_root_exact_match_counter=[]
chords_exact_counter_dups=[]
chords_exact_counter_dups2=[]
chords_exact_counter_dups3=[]
beat_counter=[]
nn=0
with open(out_json_path,'r') as out_json:
    with open(prompt_json_path,'r') as prompt_json:
        for row1,row2 in zip(prompt_json,out_json):
            a=json.loads(row1)
            b=json.loads(row2)

            #tempo
            bpm_pred=b['bpm_ext']
            features=a['bpm']
            if features is None:
                features=1000
            tempo_gt=np.sum(features>tempo_marks)
            tempo_pred=np.sum(bpm_pred>tempo_marks)
            
            if tempo_gt==tempo_pred:
                tempo_bin.append(1)
            else:
                tempo_bin.append(0)
            if np.abs(tempo_gt-tempo_pred)<2:
                tempo_bin_oneoff.append(1)
            else:
                tempo_bin_oneoff.append(0)

            bpm_gt = features
            bpm_deviation_abs.append(bpm_pred - bpm_gt)
            bpm_deviation_rel.append((bpm_pred - bpm_gt)/bpm_gt)


            key_pred=b['key2_ext'][0]
            key_pred_type=b['key2_ext'][1]
            key_gt=a['key2'][0]
            key_gt_type=a['key2'][1]

            key_pred=note_dup_list[key_pred] #convert to an index
            key_gt=note_dup_list[key_gt] #convert to an index

            if key_pred is None: #this is ok for primary only, but how to handle secondary?
                key_root_counter.append(0)
                key_type_counter.append(0)
                key_none_list.append(len(key_root_counter)-1)
                continue
            else:
                #convert to uniform representation
                if key_gt_type in maj_type_list:
                    type_gt=' major'
                elif key_gt_type in min_type_list:
                    type_gt=' minor'

                if key_pred_type in maj_type_list:
                    type_pred=' major'
                elif key_pred_type in min_type_list:
                    type_pred=' minor'

                #is root the same?
                if key_pred==key_gt:
                    key_root_counter.append(1)
                else:
                    key_root_counter.append(0)
                #is type the same?
                if type_pred==type_gt:
                    key_type_counter.append(1)
                else:
                    key_type_counter.append(0)

                #is the key the same?
                if (key_pred==key_gt) and (type_pred==type_gt):
                    key_counter.append(1)
                else:
                    key_counter.append(0)

                #and with duplicates ok?
                if (key_pred==key_gt) and (type_pred==type_gt):
                    key_counter_dup.append(1)
                elif ((np.mod(key_pred,12)==np.mod(key_gt-3,12)) and (type_pred==' major') and (type_gt==' minor')):
                    key_counter_dup.append(1)
                elif ((np.mod(key_pred-3,12)==np.mod(key_gt,12)) and (type_pred==' minor') and (type_gt==' major')):
                    key_counter_dup.append(1)
                else:
                    key_counter_dup.append(0)

            chords_gt=a['chords']
            chords_pred=b['chords_ext']

            #exact chords exact order:
            how_many = is_subsequence(chords_pred,chords_gt)
            chord_exact_match_counter.append((how_many,len(chords_gt)))

            chords_pred_copy=chords_pred.copy()
            #exact chord any order:
            chords_exact_counter_sub=0

            for ch_name in chords_gt:
                for n in range(len(chords_pred_copy)):
                    if ch_name==chords_pred_copy[n]:
                        chords_exact_counter_sub+=1
                        chords_pred_copy.pop(n)
                        break

            chords_exact_counter.append((chords_exact_counter_sub,len(chords_gt)))

            chord_root_gt=[]
            chord_root_pred=[]
            chord_type_gt=[]
            chord_type_pred=[]
            for ch in chords_gt:
                if len(ch)>1:
                    if ch[1]=='b' or ch[1]=='#': #gimme root name
                        chord_root_gt.append(ch[0:2])
                        if len(ch)>2:
                            chord_type_gt.append(ch[2:])
                        else:
                            chord_type_gt.append('major')
                    else:
                        chord_root_gt.append(ch[0])
                        chord_type_gt.append(ch[1:])
                else:
                    chord_root_gt.append(ch[0])
                    chord_type_gt.append('major')


            for ch in chords_pred:
                if len(ch)>1:
                    if ch[1]=='b' or ch[1]=='#': #gimme root name
                        chord_root_pred.append(ch[0:2])
                        if len(ch)>2:
                            chord_type_pred.append(ch[2:])
                        else:
                            chord_type_pred.append('major')
                    else:
                        chord_root_pred.append(ch[0])
                        chord_type_pred.append(ch[1:])

                else:
                    chord_root_pred.append(ch[0])
                    chord_type_pred.append('major')

            #root and exact order:
            how_many = is_subsequence(chord_root_pred,chord_root_gt)
            chord_root_exact_match_counter.append((how_many,len(chord_root_gt)))


            #exact chord any order with duplicates allowed:
            chords_exact_counter_sub=0
            #allow #b substitutions
            # extract roots and convert to single rep:
            conv_root_gt=[]
            conv_root_pred=[]
            conv_type_gt=[]
            conv_type_pred=[]
            for ch in chord_root_gt:
                conv_root_gt.append(note_dup_list[ch])
            for ch in chord_root_pred:
                conv_root_pred.append(note_dup_list[ch])
            for ch in chord_type_gt:
                if '/' in ch:
                    ch=ch.split('/')[0]
                    ch=(ch+'/')
                conv_type_gt.append(chord_type_dict[ch])
            for ch in chord_type_pred:
                if '/' in ch:
                    ch=ch.split('/')[0]
                    ch=(ch+'/')
                conv_type_pred.append(chord_type_dict[ch])


            #CMAOMM
            conv_root_pred_copy=conv_root_pred.copy()
            conv_type_pred_copy=conv_type_pred.copy()
            chords_exact_counter_sub2=0
            for ch_num,ch_typ in zip(conv_root_gt,conv_type_gt):
                for n in range(len(conv_root_pred_copy)):
                    if (ch_num==conv_root_pred_copy[n]) and (ch_typ==conv_type_pred_copy[n]): # contains minor for minor, major for major...
                        chords_exact_counter_sub2+=1
                        conv_root_pred_copy.pop(n)
                        conv_type_pred_copy.pop(n)
                        break
            chords_exact_counter_dups3.append((chords_exact_counter_sub2,len(chords_gt)))

            #exact root, any order aka are the roots there at least?
            chords_root_counter_sub=0
            for ch_name in chord_root_gt:
                for n in range(len(chord_root_pred)):
                    if ch_name==chord_root_pred[n]:
                        chords_root_counter_sub+=1
                        chord_root_pred.pop(n)
                        break
            chords_root_counter.append((chords_root_counter_sub,len(chord_root_gt)))

            #investigate number of chords
            chord_number_gt=len(chords_gt)
            chord_number_pred=len(b['chords'])
            chord_deviation_numbers.append(chord_number_pred-chord_number_gt)
            chord_gt_numbers.append(chord_number_gt)

            #only time signature, fits or doesn't fit...

            beat_gt=int(max(a['beats'][1]))
            beat_pred=int(max(b['beats_ext'][1]))

            if beat_gt==beat_pred:
                beat_counter.append(1)
            else:
                beat_counter.append(0)


#TEMPO
tempo_bin_percentage=np.round(100*tempo_bin.count(1)/(tempo_bin.count(1)+tempo_bin.count(0)),2)
tempo_bin_oneoff_percentage=np.round(100*tempo_bin_oneoff.count(1)/(tempo_bin_oneoff.count(1)+tempo_bin_oneoff.count(0)),2)


bpm_abs_mean=np.mean(np.abs(bpm_deviation_abs)).round(2)
bpm_abs_std=np.std(np.abs(bpm_deviation_abs)).round(2)

bpm_rel_mean=np.mean(np.abs(bpm_deviation_rel)*100).round(2)
bpm_rel_std=np.std(np.abs(bpm_deviation_rel)*100).round(2)

# print("Tempo bin percentage: ", tempo_bin_percentage,"Correct tempo bins: ", tempo_bin.count(1),"Wrong tempo bins: ", tempo_bin.count(0))
# print("Tempo bin one off percentage: ", tempo_bin_oneoff_percentage,"Correct tempo one off bins: ", tempo_bin_oneoff.count(1),"Wrong tempo one off bins: ", tempo_bin_oneoff.count(0))
print("Tempo bin percentage (TB): ", tempo_bin_percentage)
print("Tempo bin one off percentage (TBT): ", tempo_bin_oneoff_percentage)
# print("Bpm abs mean: ", bpm_abs_mean, "Bpm abs std: ", bpm_abs_std,"Bpm rel mean: ", bpm_rel_mean, "Bpm rel std: ", bpm_rel_std)


#KEY
key_percentage=np.round(100*key_root_counter.count(1)/(key_root_counter.count(1)+key_root_counter.count(0)),2)
key_type_percentage=np.round(100*key_type_counter.count(1)/(key_type_counter.count(1)+key_type_counter.count(0)),2)

key_root_counter2 = [key_root_counter[i] for i in range(len(key_root_counter)) if i not in key_none_list]
key_type_counter2 = [key_type_counter[i] for i in range(len(key_type_counter)) if i not in key_none_list]

key_percentage2=np.round(100*key_root_counter2.count(1)/(key_root_counter2.count(1)+key_root_counter2.count(0)),2)
key_type_percentage2=np.round(100*key_type_counter2.count(1)/(key_type_counter2.count(1)+key_type_counter2.count(0)),2)


key_exact_percentage=np.round(100*key_counter.count(1)/(key_counter.count(1)+key_counter.count(0)),2)
key_exact_percentage_dup=np.round(100*key_counter_dup.count(1)/(key_counter_dup.count(1)+key_counter_dup.count(0)),2)

print("Correct key percentage (CK): ", key_exact_percentage, "Correct key percentage with duplicates (CKD): ", key_exact_percentage_dup)

#CHORDS

# chord exact
exact_match=[]
for det,gt in chord_exact_match_counter:
    if gt==0:
        continue
    exact_match.append(det/gt)

perfect_match=100*exact_match.count(1)/len(exact_match)

exact_match_rel_mean=np.mean(100*np.array(exact_match))
exact_match_rel_std=np.std(100*np.array(exact_match))

# chord root exact
exact_match=[]
for det,gt in chord_root_exact_match_counter:
    if gt==0:
        continue
    exact_match.append(det/gt)

perfect_match_root=100*exact_match.count(1)/len(exact_match)

root_exact_match_rel_mean=np.mean(100*np.array(exact_match))
root_exact_match_rel_std=np.std(100*np.array(exact_match))

# print("Perfect chord match: ", perfect_match, "Perfect chord root match: ", perfect_match_root)

# print("Exact chord match percentage: ", exact_match_rel_mean, "Exact chord match std: ", exact_match_rel_std)
# print("Exact chord root match percentage: ", root_exact_match_rel_mean, "Exact chord root match std: ", root_exact_match_rel_std)

print("Perfect chord match (PCM): ", perfect_match)
print("Exact chord match percentage (ECM): ", exact_match_rel_mean)

#exact chords, any order...
#in how many samples all is detected, aka perfect
#do percentage per sample, average across samples
perc=[]
perfect=0
for det_ch,num_ch in chords_exact_counter:
    if num_ch==0:
        continue
    perc.append(det_ch/num_ch)
    if det_ch/num_ch==1:
        perfect+=1

perc=np.mean(perc)*100
perfect_perc=100*perfect/len(chords_exact_counter)

# print("All chords detected, any order: ", perfect_perc, "Mean percentage of detected chords in any order: ", perc)
print("All chords detected, any order (CMAO): ", perc)


#same root, any order...
# chords_root_counter=[]
perc=[]
perfect=0
for det_ch,num_ch in chords_root_counter:
    if num_ch==0:
        continue
    perc.append(det_ch/num_ch)
    if det_ch/num_ch==1:
        perfect+=1

perc=np.mean(perc)*100
perfect_perc=100*perfect/len(chords_root_counter)

# print("All roots detected, any order: ", perfect_perc, "Mean percentage of detected roots in any order: ", perc)


perc=[]
perfect=0
for det_ch,num_ch in chords_exact_counter_dups:
    if num_ch==0:
        continue
    perc.append(det_ch/num_ch)
    if det_ch/num_ch==1:
        perfect+=1

perc=np.mean(perc)*100
perfect_perc=100*perfect/len(chords_exact_counter_dups)

# print("All chords detected, any order, duplicates allowed: ", perfect_perc, "Mean percentage of detected chords in any order with dups: ", perc)

perc=[]
perfect=0
for det_ch,num_ch in chords_exact_counter_dups2:
    if num_ch==0:
        continue
    perc.append(det_ch/num_ch)
    if det_ch/num_ch==1:
        perfect+=1

perc=np.mean(perc)*100
perfect_perc=100*perfect/len(chords_exact_counter_dups2)

# print("All chords detected, any order, duplicates allowed EXPANDED: ", perfect_perc, "Mean percentage of detected chords in any order with dups EXPANDED: ", perc)

perc=[]
perfect=0
for det_ch,num_ch in chords_exact_counter_dups3:
    if num_ch==0:
        continue
    perc.append(det_ch/num_ch)
    if det_ch/num_ch==1:
        perfect+=1

perc=np.mean(perc)*100
perfect_perc=100*perfect/len(chords_exact_counter_dups3)

# print("All chords detected, any order, duplicates allowed EXPANDED_OK: ", perfect_perc, "Mean percentage of detected chords in any order with dups EXPANDED_OK: ", perc)
print('All chords detected, any order , major/minor binary distinction only (CMAOMM): ',perc)


# number of chords... percentage deviation from gt, percentage of correct number of chords...
# chord_gt_numbers=[]
# chord_deviation_numbers=[]
chord_number_mean=[]
percentage_correct=0
for i in range(len(chord_gt_numbers)):
    if chord_gt_numbers[i]==0:
        continue
    chord_number_mean.append(chord_deviation_numbers[i]/chord_gt_numbers[i])
    if chord_deviation_numbers[i]==0:
        percentage_correct+=1 
chord_number_mean=np.mean(100*np.abs(chord_number_mean)).round(2)
percentage_correct=percentage_correct*100/len(chord_gt_numbers)

#detect exact ones
#detect partial ones - keep the percentage
# print("Correct number of chords percentage: ", percentage_correct, "Mean relative difference of chord number: ", chord_number_mean)

#BEATS
beat_percentage=np.round(100*beat_counter.count(1)/(beat_counter.count(1)+beat_counter.count(0)),2)
# print("Correct beat percentage: ", beat_percentage, "Correct beats: ", beat_counter.count(1),"Wrong beats: ", beat_counter.count(0))
print("Correct beat percentage (BC): ", beat_percentage)






