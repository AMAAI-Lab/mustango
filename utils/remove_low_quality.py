
import json
from glob import glob
import os
import random
import numpy as np


old_json_path='/music-caps/musiccaps.json'
new_json_path='/music-caps/MC3L.json'

data_len=5479

mode="keep" #keep or remove

with open(new_json_path, 'w') as json_out: #out

    with open(old_json_path, 'r') as json_in:
        k=0
        for row in json_in:
            a=json.loads(row)
            caps=a['captions']

            forbidden=['low quality','low-quality','low fidelity','low-fidelity','quality is poor','quality is low','poor quality'] #inferior, not great, bad... other words... let's remove "quality"...

            if 'quality' in caps.lower() and 'poor' in caps.lower():
                if mode=="remove":
                    continue
                elif mode=="keep":
                    json_out.write(json.dumps(a) + '\n')
                    continue

            alarm=False
            for forbi in forbidden:
                if forbi in caps.lower():
                    alarm=True
                    break
            if alarm==True:
                if mode=="remove":
                    continue
                elif mode=="keep":
                    json_out.write(json.dumps(a) + '\n')
                    continue

            if 'quality' in caps.lower(): #just remove quality, it always is related to "bad", "poor", or "average" at maximum...
                if mode=="remove":
                    continue
                elif mode=="keep":
                    json_out.write(json.dumps(a) + '\n')
                    continue

            if mode=="remove":
                json_out.write(json.dumps(a) + '\n')
            k=k+1
            print(k)
