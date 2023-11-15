import json
import os
import numpy as np

infile='/666/TANGO/music-caps/MC_augmented.json'
outfile='/666/TANGO/music-caps/MC_augmented_text_expanded.json'

all_prompts=False

with open(outfile,'w') as outjson:
    with open(infile,'r') as injson:
        for row in injson:
            a=json.loads(row)
            new_row=a.copy()
            if all_prompts:
                how_many_sentences=4
            else:
                how_many_sentences=np.random.choice(5,p=np.array((0.15,0.25,0.3,0.2,0.1)))


            prompt_list=[new_row['prompt_ch'], new_row['prompt_bt'], new_row['prompt_bpm'], new_row['prompt_key2']]
            music_prompts=''
            rand_ind=np.random.permutation(4)
            for i in range(how_many_sentences):
                music_prompts=music_prompts + ' ' + prompt_list[rand_ind[i]]
            music_prompts=music_prompts + ' ' + a['prompt_aug']
            music_prompts = music_prompts.replace('  ',' ', 3)
            new_row['caption']=(new_row['caption'] + ' ' + music_prompts).replace('  ',' ',3)
            outjson.write(json.dumps(new_row) + '\n')
