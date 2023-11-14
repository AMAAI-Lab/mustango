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
import librosa
from utils.keyfinder import Tonal_Fragment

import pyrubberband as pyrb


class sine_creator(object):
	def __init__(self, dur, sr, amp=None):

		"""
		gen = sine_creator(dur = 2, sr = 16000)
		gen([440, 880, 220])
		"""
		self.dur = dur
		self.sr = sr
		self.amp = amp #between [0,1]
	def __call__(self, freqs = []):
		t = np.linspace(0., 1., int(self.dur*self.sr))
		if self.amp is None:
			self.amp = 1/len(freqs)
		sins = sum([self.amp*np.sin(2. * np.pi * f * t) for f in freqs])
		# name = "_".join([str(x) for x in freqs])+".wav"
		# write(name, self.sr, sins.astype(np.float32))

		return sins.astype(np.float32)

def create_click_track(sr, click_timing = [1, 2, 3, 4], click_beat = [1, 2, 3, 4, 1, 2], total_time = 5, impulse_dur = 0.1):
	click_track = np.zeros(int(np.ceil(total_time*sr)))
	downbeat = sine_creator(dur = impulse_dur, sr = sr, amp = 0.5)
	downbeat_click = downbeat([440, 220, 110])


	beat_ = sine_creator(dur = impulse_dur, sr = sr, amp = 0.25)
	beat_click = beat_([330, 110, 55])

	for click, beat in zip(click_timing, click_beat):

		if beat == 1:
			if click !=0:
				tmp_click = np.concatenate( (np.zeros(int((click - impulse_dur)*sr)),  downbeat_click,  np.zeros(int((total_time - click)*sr))  ) )
			else:
				tmp_click = np.concatenate( ( downbeat_click,  np.zeros(int((total_time - impulse_dur)*sr))  ) )
		else: 
			if click!=0:
				tmp_click = np.concatenate( (np.zeros(int((click - impulse_dur)*sr)),  beat_click,  np.zeros(int((total_time - click)*sr))  ) )
			else:
				tmp_click = np.concatenate( ( beat_click,  np.zeros(int((total_time - impulse_dur)*sr))  ) )

		#pad
		if len(tmp_click)>len(click_track):
			tmp_click = tmp_click[:len(click_track)]
		elif len(tmp_click)<len(click_track):
			tmp_click = np.concatenate((tmp_click, np.zeros(len(click_track)-len(tmp_click))))
		
		click_track += tmp_click

	return click_track

class BeatProcessor(object):
	def __init__(self, estimator, if_aux_click, aux_click_save_path):
		self.estimator = estimator
		self.if_aux_click = if_aux_click
		self.aux_click_save_path = aux_click_save_path
	def __call__(self, path):
		Output = self.estimator.process(path)
		click_timing, click_beat = Output[:, 0], Output[:, 1]	

		if self.if_aux_click and self.aux_click_save_path:
			os.makedirs(self.aux_click_save_path, exist_ok=True)

			input_file, sr = sf.read(path)
			total_time = len(input_file)/sr
			click_track = create_click_track(sr, impulse_dur = 0.02, click_timing = click_timing, click_beat = click_beat, total_time = total_time)
			name = path.split("/")[-1][:-4]+"click.wav"
			write(f"{self.aux_click_save_path}/{name}", sr, (click_track+input_file).astype(np.float32))

		
		return click_timing, click_beat #each is a numpy array with an arbitrary len

class ChordProcessor(object):
	def __init__(self, estimator, if_process_group):
		self.estimator = estimator
		self.if_process_group = if_process_group
	def post_proc(self, chords):
		if self.if_process_group:
			return [ (x[0], [ (y.chord, y.timestamp) for y in x[1][1:-1]]) for x in chords] #chord output always return 10.309659863 as the last entry and 0.371519274 as the first entry --> trim
			
		else:

			return [ (x.chord, x.timestamp) for x in chords[1:-1]]

	def __call__(self, path_or_paths):
		if self.if_process_group:
			chords = self.estimator.extract_many(path_or_paths, num_extractors=2, num_preprocessors=2, max_files_in_cache=10, stop_on_error=False)
		else:
			chords = self.estimator.extract(path_or_paths)
		
		

		return self.post_proc(chords)

def get_bpm(beats):
	#get diff
	if len(beats[0])<3: #too little beats to determine bpm
		return None
	diff=np.diff(beats[0])
	loc_bpm=1/diff*60
	#median filter the diff
	loc_bpm2=np.concatenate((np.array(loc_bpm[0]).reshape(1),loc_bpm,np.array(loc_bpm[-1]).reshape(1)))
	for i in range(len(loc_bpm2)-1):
		loc_bpm2[i+1]=np.median(loc_bpm2[i:i+3])
	loc_bpm=loc_bpm2[1:-1]
	avg_bpm = np.round(np.mean(loc_bpm))
	# return (bpm, markers), avg_bpm
	return avg_bpm

def get_key(file_path):
	y, sr = librosa.load(file_path,sr=None)
	y_harmonic, y_percussive = librosa.effects.hpss(y)
	analyzed_seg = Tonal_Fragment(y_harmonic, sr)
	# analyzed_seg.print_chroma()
	# analyzed_seg.print_key()
	# analyzed_seg.corr_table()
	key, corr, altkey, altcorr = analyzed_seg.give_key()
	return key, corr, altkey, altcorr

def format_key(key_list):
	if key_list==None:
		return None
	out_list=[]
	blist=['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
	slist=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
	maj_list=['maj',' maj','M','major',' major']
	min_list=['m',' min','min','minor',' minor']

	for keys in key_list:
		if keys is None:
			out_list.append(None)
		else:
			key_root, key_type = keys.split(' ')
			if 'major' in key_type:
				index=np.random.randint(0,5)
				new_type = maj_list[index]
			elif 'minor' in key_type:
				index=np.random.randint(0,5)
				new_type = min_list[index]
			# convert root too? # to b and b to #?
			if np.random.randint(0,10)==0: #10 percent of cases, change b for # and vice versa
				if '#' in key_root:
					key_root=blist[slist.index(key_root)]
				elif 'b' in key_root:
					key_root=slist[blist.index(key_root)]	
			out_list.append(key_root+new_type)
	return out_list


def speed_change(audio_samples, sr, speed_rate_start_end_triplets):
	speed_rate_start_end_triplets_filled = [(1.0, 0, 0)]
	for rate, start, end in speed_rate_start_end_triplets:
		last_end_time = speed_rate_start_end_triplets_filled[-1][-1]
		if start != last_end_time:
			speed_rate_start_end_triplets_filled.append((1.0, last_end_time, start))
		speed_rate_start_end_triplets_filled.append((rate, start, end))

	tmp = []
	for rate, start, end in speed_rate_start_end_triplets_filled:
		audio_chunk = audio_samples[int(start * sr): int(end * sr)]
		if rate!=1.0:
			audio_chunk_stretched = pyrb.time_stretch(audio_chunk, sr, rate = rate)
		else:
			audio_chunk_stretched = audio_chunk
		tmp.append(audio_chunk_stretched)

	#calculate timestamps where tempo changes in the augmented audio
	per_step_change = [  float(x[2]-x[1])*(1 - 1/x[0])  for x in speed_rate_start_end_triplets]
	delta_durs = np.cumsum([0.]+per_step_change)[:-1]
	adjusted_speed_rate_start_end_triplets = [ (x[0], x[1] - delta_dur, x[2]-delta_dur - per_step)    for x, per_step, delta_dur in zip(speed_rate_start_end_triplets, per_step_change, delta_durs)]

	return np.concatenate(tmp), adjusted_speed_rate_start_end_triplets

def pitch_change(audio_samples, sr, pitch_steps_start_end_triplets):
	pitch_steps_start_end_triplets_filled = [(0, 0, 0)]
	for rate, start, end in pitch_steps_start_end_triplets:
		last_end_time = pitch_steps_start_end_triplets_filled[-1][-1]
		if start != last_end_time:
			pitch_steps_start_end_triplets_filled.append((0, last_end_time, start))
		pitch_steps_start_end_triplets_filled.append((rate, start, end))

	tmp = []
	for steps, start, end in pitch_steps_start_end_triplets_filled:
		audio_chunk = audio_samples[int(start * sr): int(end * sr)]
		if steps!=0:
			audio_chunk_stretched = pyrb.pitch_shift(audio_chunk, sr, n_steps = steps)
		else: 
			audio_chunk_stretched = audio_chunk
		tmp.append(audio_chunk_stretched)

	return np.concatenate(tmp)

def volume_change(audio_samples, sr, volume_amp_start_end_triplets):
	volume_amp_start_end_triplets_filled = [(1., 0, 0)]
	for rate, start, end in volume_amp_start_end_triplets:
		last_end_time = volume_amp_start_end_triplets_filled[-1][-1]
		if start != last_end_time:
			volume_amp_start_end_triplets_filled.append((1., last_end_time, start))
		volume_amp_start_end_triplets_filled.append((rate, start, end))
	
	tmp = []
	for rate, start, end in volume_amp_start_end_triplets_filled:
		audio_chunk = audio_samples[int(start * sr): int(end * sr)]
		if rate!=1.0:
			audio_chunk_stretched = audio_chunk*rate
		else: 
			audio_chunk_stretched = audio_chunk
		tmp.append(audio_chunk_stretched)

	return np.concatenate(tmp)

def volume_change_simple(audio_samples, sr, volume_amp_start_end_quadruple, expo=True):
	#expects only one change - either from the start to somewhere, or from middle until the end
	rate_start, rate_end, start, end = volume_amp_start_end_quadruple[0]
	sample_len=len(audio_samples)
	if start==0:
		if expo:
			# multiplier = np.concatenate((10**(np.linspace(np.log10(rate_start)/np.log10(10),np.log10(rate_end)/np.log10(10),(np.ceil(end*sr)-np.floor(start*sr)).astype(int)[0])),np.ones(np.floor(sample_len-end*sr).astype(np.int))*rate_end))
			multiplier = np.concatenate((10**(np.linspace(np.log10(rate_start)/np.log10(10),np.log10(rate_end)/np.log10(10),int(np.ceil(end*sr)-np.floor(start*sr)))),np.ones(int(np.floor(sample_len-end*sr)))*rate_end))
		else:
			multiplier = np.concatenate((np.linspace(rate_start,rate_end,int(np.ceil(end*sr)-np.floor(start*sr))),np.ones(int(np.floor(sample_len-end*sr)))*rate_end))
	else:
		if expo:
			multiplier = np.concatenate((np.ones(int(np.floor(start*sr)))*rate_start,10**(np.linspace(np.log10(rate_start)/np.log10(10),np.log10(rate_end)/np.log10(10),int(np.ceil(end*sr)-np.floor(start*sr))))))
		else:
			multiplier = np.concatenate((np.ones(int(np.floor(start*sr)))*rate_start,np.linspace(rate_start,rate_end,int(np.ceil(end*sr)-np.floor(start*sr)))))
	y = audio_samples * multiplier

	return y

def chord_shift(chord,shift):

	if chord=='N':
		return 'N'

	blist=['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
	slist=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
	# blistsmall=['Db','Eb','Gb','Ab','Bb']
	# slistsmall=['C#','D#','F#','G#','A#']
	# indlist=[1,3,6,8,10]
	#split by / for enhanced chords
	schords=chord.split('/')
	out_chord=''
	i=-1
	for sch in schords:
		i+=1
		if i>0: # we successfully split by '/', need to add it back to the string
			out_chord=out_chord+'/'
		if len(sch)>1: # this part might have a '#' or 'b'
			if sch[1]=='#' or sch[1]=='b':
				name=sch[0:2]
			else:
				name=sch[0]
		else:
			name=sch[0]

		sch_s=sch.split(name)
		if 'b' in name:
			if name=='Fb':
				name='E'
			elif name=='Cb':
				name='B'
			new_ind = np.mod(blist.index(name)+shift,len(blist)) #cycle through root note list
			new_name = blist[new_ind]
		else: # '#' in name, or none
			if name=='E#':
				name='F'
			elif name=='B#':
				name='C'
			new_ind = np.mod(slist.index(name)+shift,len(slist)) #cycle through root note list
			new_name = slist[new_ind]

		out_chord = out_chord + new_name + sch_s[1]#keep and concat
	return out_chord


def chord_pitch_change_fc(chords,triplet):
	if len(chords)<1:
		return chords
	# shift, start, end = triplet
	if triplet[0][1]==0:
		new_ch=[]
		shift=triplet[0][0]
		for ch in chords:
			new_ch.append([chord_shift(ch[0],shift),ch[1]])
	else:
		first = True
		for shift, start, end in triplet:
			new_ch=[]
			i=-1
			for ch in chords:
				i+=1
				if start > ch[1]: #no change yet!
					new_ch.append([ch[0],ch[1]])
					continue
				# elif ch[1]<0.47: #the chord is there from the very start (chord detector induces a shift on the timestamps)
				# 	#shift everything, don't duplicate
				# 	first = False
				# 	new_ch.append([chord_shift(ch[0],shift),ch[1]])
				else: # after shift, change it
					if first: #duplicate and shift
						new_ch.append([chord_shift(chords[i-1][0],shift),start[0]]) #duplication
						first = False
						new_ch.append([chord_shift(ch[0],shift),ch[1]])
					else: #shift only
						new_ch.append([chord_shift(ch[0],shift),ch[1]])

	return new_ch

def beat_change_fc(beats,triplets,crop=10):
	# rate, start, end = triplet
	for rate, start, end in triplets: #change the feature for every triplet
		i=0
		for b in beats[0]: #cycle through the single beats to find where to start from
			if b<start:
				i+=1
				continue
			else:
				beats[0][i:]=((beats[0][i:]-start)/rate)+start
				break
	i=0
	if beats[0][-1]>crop: #need to crop?
		for b in beats[0]: #scan for cropping
			if b<crop: #crop the feature so that it matches the max audio length specified by "crop"
				i+=1
			else:
				beats=[beats[0][:i],beats[1][:i]]
	return beats

def chord_speed_change_fc(chords,triplets,crop=10):
	# rate, start, end = triplet
	if len(chords)<1:
		return chords
	for rate, start, end in triplets: #change the feature for every triplet
		i=0
		for chord,time_stamp in chords: #cycle through the single beats to find where to start from
			if time_stamp<start:
				i+=1
			else:
				chords[i][1]=float(((chords[i][1]-start)/rate)+start)
				i+=1
				# beats[0][i:]=((beats[0][i:]-start)/rate)+start
	i=0
	if chords[-1][1]>crop: #need to crop?
		for ch,ts in chords: #scan for cropping
			if ts<crop: #crop the feature so that it matches the max audio length specified by "crop"
				i+=1
			else:
				chords=chords[0:i]

	return chords


def bpm_change_fc(bpm,triplets):
	if bpm is None:
		return None
	if len(triplets)==1:
		bpm = bpm * triplets[0][0]
		return np.round(bpm)
	else:
		return None

def key_change_fc(key_list,triplets):
	if len(triplets)==1:
		shift=triplets[0][0]
		blist=['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
		slist=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
		new_keys=[]
		for keys in key_list:
			if keys is None:
				new_keys.append(None)
			else:
				key_root, key_type = keys.split(' ')
				if 'b' in key_root:
					new_ind = np.mod(blist.index(key_root)+shift,len(blist)) #cycle through root note list
					new_root = blist[new_ind]
				else:
					new_ind = np.mod(slist.index(key_root)+shift,len(slist)) #cycle through root note list
					new_root = slist[new_ind]			
				new_keys.append(new_root + ' ' + key_type)
			#shift the key through a loop
			#TODO
	else:
		return None

	return new_keys


if __name__ == '__main__':

	config_path = "/666/TANGO/gh/text2music/config/music_feat_extrator_config.yaml"
	#config 
	with open (config_path, 'r') as f:
		cfg = yaml.safe_load(f)
	
	beat_estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=["beat_particles"], thread=False)
	beat_processor = BeatProcessor(beat_estimator, **cfg['beat_processor'])
	
	chord_estimator = Chordino()  
	chord_processor = ChordProcessor(chord_estimator,**cfg['chord_processor'])

	paths = glob.glob(cfg['data_path']+"/*.wav")[:2]
	
	#detect beat and chords
	chord_feat = chord_processor(paths)
	print(chord_feat)

	for path in paths:
		beat_feat = beat_processor(path)
		print(beat_feat)

	speed_rate_start_end_triplets = [(1.5, 1, 2), (0.5, 3, 6), (2.0, 8, 10)]
	pitch_steps_start_end_triplets = [(1, 1, 3), (-2, 4, 6), (3, 8, 10)] #"from 0-3secs, 1.5 the speed"
	volume_amp_start_end_triplets = [(0.5, 0, 0.2), (0.6, 0.2, 0.4), (0.7, 0.4, 0.6), (0.8, 0.6, 0.8), (1.0, 0.8, 1.0), (1.2, 1.0, 1.2), (1.4, 1.2, 1.4), (1.6, 1.4, 1.6), (1.8, 1.6, 1.8), (3.0, 1.8, 10) ]

	y, sr = sf.read(paths[1])
	y_speed_change, adjusted_speed_rate_start_end_triplets = speed_change(y, sr, speed_rate_start_end_triplets)
	y_pitch_change = pitch_change(y, sr, pitch_steps_start_end_triplets)
	y_volume_chamge = volume_change(y, sr, volume_amp_start_end_triplets)
	write("speed_change_tmp.wav", sr, y_speed_change.astype(np.float32))
	write("pitch_change_tmp.wav", sr, y_pitch_change.astype(np.float32))
	write("volume_change_tmp.wav", sr, y_volume_chamge.astype(np.float32))
