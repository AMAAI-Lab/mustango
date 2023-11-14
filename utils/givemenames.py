import os
from glob import glob
import json
import shutil

# output_path='data/musiccaps_test_references'
# data_path='/666/TANGO/music-caps/data'
ref_file='/666/TANGO/tango/data/test_musiccaps.json'
text_file='/666/TANGO/tango/data/output_captions.txt'

with open(ref_file, encoding='utf-8') as ref:
    i=0
    with open (text_file, 'w', encoding='utf-8') as txt:

        for row in ref:
            a=json.loads(row)
            caption=a['captions']
            txt.write(str(i)+' '+caption+' \n')
            # shutil.copy(a['location'],os.path.join(output_path,'output_{}.wav'.format(i)))

            print(row)
            i+=1