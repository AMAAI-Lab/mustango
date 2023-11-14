import json
import numpy as np

json_to_investigate='/666/TANGO/music-caps/train_musiccaps_ep.json'


chords=[]
chord_stamps=[]
beat_type=[]
with open(json_to_investigate, encoding='utf-8') as jsi:
    new_row={}
    for row in jsi:
        a=json.loads(row)
        b=a['chords']
        for ch,chs in b:
            chords.append(ch)
            chord_stamps.append(chs)
        beats=a['beats']
        beat_type.append(np.max(beats[1]))


print(set(beat_type))
# print(chords)
# print(chord_stamps)

stamps=sorted(set(chord_stamps))
all_c=set(chords)
how_many_c=[]
for ele in all_c:
    how_many_c.append(chords.count(ele))

how_many_s=[]
for ele in stamps:
    how_many_s.append(chord_stamps.count(ele))
# how_many_c=chords.count()

chord_types=[]
chord_types.append(None)
for ch in chords:
    if len(ch)>1:
        if ch[1]=='#' or ch[1]=='b':
            root_name=ch[0:2]
        else:
            root_name=ch[0]
    else:
        root_name=ch[0]
    ch_s = ch.split(root_name)
    if ch_s[1]=='':
        chord_types.append('major')
        continue
    else:
        if '/' in ch_s[1]:
            ch_ss=ch_s[1].split('/')
            chord_types.append(ch_ss[0]+'/')
        else:
            chord_types.append(ch_s[1])


print(chord_types)
xx=set(chord_types)
print(xx)