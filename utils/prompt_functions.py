import numpy as np

def speed_prompt_fc(triplets,sr,crop=10):
	# assuming that end of one tempo phase starts with another tempo phase or the end of the song
	num_speed_changes=len(triplets)
	# if len(triplets)==1:
	# 	tempo_scale, speed_timestamps, speed_ends = triplets[0]
	# elif len(triplets)==2:
	# 	tempo_scale[0], speed_timestamps[0], speed_ends[0] = triplets[0]
	# 	tempo_scale[1], speed_timestamps[1], speed_ends[1] = triplets[1]

	tempo_scale, speed_timestamps, speed_end = np.zeros(len(triplets)), np.zeros(len(triplets)), np.zeros(len(triplets))
	for i in range(len(triplets)):
		tempo_scale[i], speed_timestamps[i], speed_end[i] = triplets[i]
	speed_timestamps=np.round(speed_timestamps,2)
	speed_ends=np.round(speed_end,2)
	#single aug case only
	for i in range(len(triplets)):
		if speed_ends[i]>crop:
			speed_ends[i]=np.array([crop])
		if speed_timestamps[i]>crop:
			speed_timestamps[i]=np.array([crop])
	tempo_text=''
	if num_speed_changes==1:
		#roll a dice to determine which to use
		dice=np.random.randint(0,4) # can go octave
		if dice==0:
			tempo_text+=("The song's tempo is changed by a factor of {} after {} seconds.".format(tempo_scale[0],speed_timestamps[0]))
		elif dice==1:
			tempo_text+=("At {} seconds into the song, the tempo changes {} times.".format(speed_timestamps[0],tempo_scale[0]))
		elif dice==2:
			if tempo_scale[0]>1:
				tempo_text+=("The song is sped up by a factor of {} at {} seconds.".format(tempo_scale[0],speed_timestamps[0]))
			else:
				tempo_text+=("The song is slowed down by a factor of {} at {} seconds.".format(tempo_scale[0],speed_timestamps[0]))
		else:
			if (tempo_scale[0]>1)&(tempo_scale[0]<1.5):
				tempo_text+=("Midway through, the song is slightly sped up.")
			elif (tempo_scale[0]>1.5):
				tempo_text+=("Midway through, the tempo of the song jumps up.")
			elif (tempo_scale[0]>0.75)&(tempo_scale[0]<1):
				tempo_text+=("In the middle, the song is slowed down a little.")
			else:
				tempo_text+=("Midway through the tempo drops down significantly.")
				
	elif num_speed_changes==2:
		dice=np.random.randint(0,3)
		if dice==0:
			tempo_text+=("The song's tempo is first changed by a factor of {} after {} seconds. Then, the tempo is changed by a factor of {} after additional {} seconds.".format(tempo_scale[0],speed_timestamps[0],tempo_scale[1],speed_timestamps[1]))
		elif dice==1:
			tempo_text+=("At {} seconds into the song, the tempo changes {} times. Later, after {} more seconds, the tempo is changed by a factor of {}.".format(speed_timestamps[0],tempo_scale[0],speed_timestamps[1],tempo_scale[1]))
		else:
			if tempo_scale[0]>1:
				tempo_text+=("The song is sped up by a factor of {} at {} seconds. ".format(tempo_scale[0],speed_timestamps[0]))
			else:
				tempo_text+=("The song is slowed down by a factor of {} at {} seconds. ".format(tempo_scale[0],speed_timestamps[0]))
			if tempo_scale[1]>1:
				tempo_text+=("After {} more seconds, the song is sped up by a factor of {}.".format(speed_timestamps[0],tempo_scale[0]))
			else:
				tempo_text+=("After {} more seconds, the song is slowed down by a factor of {}.".format(speed_timestamps[0],tempo_scale[0]))
	return tempo_text

def pitch_prompt_fc(triplets,sr):
	# PITCH PROMPTS
	pitch_text=''
	#option A - general formula
	# for i in range(num_speed_change):
	# 	tempo_text.append("The song's tempo is changed by a factor of {}.".format(tempo_scale(i)))

	#option B - manual for our case, might be better
	pitch_shift, pitch_timestamps, pitch_end = np.zeros(len(triplets)), np.zeros(len(triplets)), np.zeros(len(triplets))
	for i in range(len(triplets)):
		pitch_shift[i], pitch_timestamps[i], pitch_end[i] = triplets[i]
	pitch_timestamps=np.round(pitch_timestamps,2)
	pitch_end=np.round(pitch_end,2)
	pitch_shift=np.array(pitch_shift.astype(np.int))
	# pitch_shift, pitch_timestamps, pitch_end = triplets[0] #single triplet case only
	if len(triplets)==1:
		if pitch_shift[0]==12: #octave case
			dice=np.random.randint(0,2)
			if dice==0:
				pitch_text+=("The song is transposed up by an octave after {} seconds.".format(pitch_timestamps[0]))
			elif dice==1:
				pitch_text+=("Midway through, the song goes up by an octave.")
		elif pitch_shift[0]==-12: #octave case
			dice=np.random.randint(0,2)
			if dice==0:
				pitch_text+=("The song is transposed down by an octave after {} seconds.".format(pitch_timestamps[0]))
			elif dice==1:
				pitch_text+=("Midway through, the song goes down by an octave.")
		else: #non-octave cases
			dice=np.random.randint(0,3) #roll a dice to determine which to use
			if dice==0:
				pitch_text+=("The song's pitch is changed by {} semitones after {} seconds.".format(pitch_shift[0],pitch_timestamps[0]))
			elif dice==1:
				if np.mod(pitch_shift[0],2)==0:
					pitch_text+=("At {} seconds into the song, the song is transposed by {} full tones.".format(pitch_timestamps[0],pitch_shift[0]/2))
				else:
					pitch_text+=("At {} seconds into the song, the song is transposed by {} semitone steps.".format(pitch_timestamps[0],pitch_shift[0]))
			else:
				if pitch_shift[0]>0: #say up or down
					if np.mod(pitch_shift[0],2)==0: #full tone?
						pitch_text+=("The song is transposed up by {} tones at {} seconds.".format(pitch_shift[0]/2,pitch_timestamps[0]))
					else:
						pitch_text+=("The song is transposed up by {} steps at {} seconds.".format(np.abs(pitch_shift[0]),pitch_timestamps[0]))
				else:
					if np.mod(pitch_shift[0],2)==0: #full tone?
						pitch_text+=("The song is transposed down by {} tones at {} seconds.".format(np.abs(pitch_shift[0])/2,pitch_timestamps[0]))
					else:
						pitch_text+=("The song is transposed down by {} steps at {} seconds.".format(np.abs(pitch_shift[0]),pitch_timestamps[0]))

	elif len(triplets)==2:
		dice=np.random.randint(0,3)
		if dice==0:
			pitch_text+=("The song's tempo is first changed by a factor of {} after {} seconds. Then, the tempo is changed by a factor of {} after additional {} seconds.".format(pitch_shift[0],pitch_timestamps[0],pitch_shift[1],pitch_timestamps[1]))
			# tempo_text+=("The song's tempo is changed by a factor of {} after {} seconds.".format(tempo_scale[0],speed_timestamps[0]/sr))
		elif dice==1:
			pitch_text+=("At {} seconds into the song, the tempo changes {} times. Later, after {} more seconds, the tempo is changed by a factor of {}.".format(pitch_timestamps[0],pitch_shift[0],pitch_timestamps[1],pitch_shift[1]))
			# tempo_text+=("At {} seconds into the song, the tempo changes {} times.".format(speed_timestamps[0]/sr,tempo_scale[0]))
		else:
			if pitch_shift[0]>0:
				pitch_text+=("The song is transposed up by {} semitones at {} seconds. ".format(pitch_shift[0],pitch_timestamps[0]))
			else:
				pitch_text+=("The song is transposed down by {} semitones at {} seconds. ".format(np.abs(pitch_shift[0]),pitch_timestamps[0]))
			if pitch_shift[1]>0:
				pitch_text+=("After {} more seconds, the song is transposed up by {} semitones.".format(pitch_timestamps[0],pitch_shift[0]))
			else:
				pitch_text+=("After {} more seconds, the song is transposed down by {} semitones.".format(pitch_timestamps[0],np.abs(pitch_shift[0])))
	return pitch_text



def cres_prompt_fc(quadruple,sr):
	if len(quadruple)==1:
		rate_start, rate_end, cres_start, cres_end=quadruple[0]
	cres_text=''
	cres_end=np.round(cres_end,2)
	cres_start=np.round(cres_start,2)
	# split by strength of cres and decres
	if cres_start==0: # cres/decres at the start
		dice=np.random.randint(0,3)
		if rate_start<1: #crescendo
			if dice==0:
				cres_text+=("There is a crescendo from start until {} seconds.".format(cres_end))
			elif dice==1:
				cres_text+=("The song starts with a crescendo.")
			else:
				cres_text+=("Increase the volume progressively!")

		else: #decrescendo
			if dice==0:
				cres_text+=("There is a decrescendo from start until {} seconds.".format(cres_end))
			elif dice==1:
				cres_text+=("The song starts with a decrescendo.")
			else:
				cres_text+=("Decrease the volume progressively!")

	else: #cres/descres midway through
		dice=np.random.randint(0,4)
		if rate_start<1: #crescendo
			if dice==0:
				cres_text+=("There is a crescendo from {} seconds on.".format(cres_start))
			elif dice==1:
				cres_text+=("At seconds {}, the song starts to gradually increase in volume.".format(cres_start))
			elif dice==2:
				cres_text+=("Midway through the song, a crescendo starts.")
			else:
				cres_text+=("Increase the volume progressively!")

		else: #decrescendo
			if dice==0:
				cres_text+=("There is a decrescendo from {} seconds on.".format(cres_start))
			elif dice==1:
				# cres_text+=("At seconds {}, the song starts to decrease in volume.".format(cres_timestamps[0]/sr))
				cres_text+=("After a few seconds, the song starts to decrease in volume.")
			elif dice==2:
				cres_text+=("Midway through the song, a decrescendo starts.")
			else:
				cres_text+=("Decrease the volume progressively!")
	return cres_text


def chords_prompt_fc(chords):
	if not chords:
		return ''
	dice=np.random.randint(0,2)
	if dice==0:
		cho=("The chord sequence is")
	elif dice==1:
		cho=("The chord progression in this song is")

	for ch in chords:
		cho=cho+(" {},".format(ch[0]))
	cho=cho[:-1]+"."
	return cho

def beats_prompt_fc(beats):
	if not beats:
		return ''
	if not beats[0].shape[0]:
		return ''
	elif beats[0].shape[0]==0:
		return ''
	elif beats[0].shape[0]==1:
		return ''
	timestamps, beat = beats
	dice=np.random.randint(0,3)
	# beats_prompt=[]
	# if max seen number is...
	if dice==0:
		# beats_prompt.append("The time signature is {}.".format(np.int(np.max(beat)))) #.format max(beats)?
		beats_prompt=("The time signature is {}/4.".format(np.int(np.max(beat)))) #.format max(beats)?
	# beats_prompt.append("The time signature is {}/4.".format(np.int(np.max(beat))))#?
	elif dice==1:
		beats_prompt=("The beat is {}.".format(np.int(np.max(beat))))
	elif dice==2:
		beats_prompt=("The beat counts to {}.".format(np.int(np.max(beat))))
	else:
		beats_prompt=("Downbeat is every {} beats.".format(np.int(np.max(beat))))

	return beats_prompt

def bpm_prompt_fc(bpm):
	if not bpm:
		return ''
	which_type=np.random.randint(0,2)
	if which_type==0:
		dice=np.random.randint(0,3)
		if dice==0:
			bpm_prompt=("The bpm is {}.".format(bpm))
		elif dice==1:
			bpm_prompt=("The tempo of this song is {} beats per minute.".format(bpm))
		elif dice==2:
			bpm_prompt=("This song goes at {} beats per minute.".format(bpm))
	else:
		tempo_marks=np.array((40, 60, 70, 90, 110, 140, 160, 210))
		tempo_caps=['Grave', 'Largo', 'Adagio', 'Andante', 'Moderato', 'Allegro', 'Vivace', 'Presto', 'Prestissimo']
		index=np.sum(bpm>tempo_marks)
		cap=tempo_caps[index]
		dice=np.random.randint(0,4)
		if dice==0:
			bpm_prompt=("This song is in {}.".format(cap))
		elif dice==1:
			bpm_prompt=("The tempo of this song is {}.".format(cap))
		elif dice==2:
			bpm_prompt=("This song is played in {}.".format(cap))
		elif dice==3:
			bpm_prompt=("The song is played at the pace of {}.".format(cap))

	return bpm_prompt

def key_prompt_fc(key_list):
	if not key_list:
		return ''
	key, altkey = key_list
	if key==None:
		return ''

	if altkey==None:
		dice=np.random.randint(0,3)
		if dice==0:
			key_prompt=("The key is {}.".format(key))
		elif dice==1:
			key_prompt=("The key of this song is {}.".format(key))
		elif dice==2:
			key_prompt=("This song is in the key of {}.".format(key))

	elif altkey is not None:
		dice=np.random.randint(0,3)
		if dice==0:
			key_prompt=("The key is {}, or {}.".format(key,altkey))
		elif dice==1:
			key_prompt=("This song is in the key of {} or {}.".format(key,altkey))
		else:
			key_prompt=("This piece is either in the key of {} or {}.".format(key,altkey))

	return key_prompt