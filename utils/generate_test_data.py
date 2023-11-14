import os
from glob import glob
import json
import shutil

output_path='data/100musiccaps_test_references'
data_path='/666/TANGO/music-caps/data'
ref_file='/666/TANGO/tango/data/100test_musiccaps_orig.json'


with open(ref_file, encoding='utf-8') as ref:
    i=0
    for row in ref:
        a=json.loads(row)
        shutil.copy(a['location'],os.path.join(output_path,'output_{}.wav'.format(i)))

        print(row)
        i+=1

# f = open(ref_file)

# data=json.load(f)


# print('yolo')