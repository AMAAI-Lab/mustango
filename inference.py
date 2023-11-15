import os
import copy
import json
import time
import torch
import argparse
import soundfile as sf
import wandb
from tqdm import tqdm
from diffusers import DDPMScheduler
from audioldm_eval import EvaluationHelper
from models import build_pretrained_models, AudioDiffusion, MusicAudioDiffusion
from transformers import AutoProcessor, ClapModel
import torchaudio
from tango import Tango
import numpy as np

class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	
def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]

def parse_args():
	parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
	parser.add_argument(
		"--original_args", type=str, default="saved/trinity160epo-MC3/summary.jsonl",
		help="Path for summary jsonl file saved during training."
	)
	parser.add_argument(
		"--model", type=str, default="saved/trinity160epo-MC3/epoch_160/pytorch_model_2.bin",
		help="Path for saved model bin file."
	)
	parser.add_argument(
		"--test_file", type=str, default="data/MC3_testBfinal_preds2.json",
		help="json file containing the test prompts for generation."
	)
	parser.add_argument(
		"--text_key", type=str, default="captions",
		help="Key containing the text in the json file."
	)
	parser.add_argument(
		"--test_references", type=str, default="data/MC3_testA_references",
		help="Folder containing the test reference wav files."
	)
	parser.add_argument(
		"--num_steps", type=int, default=200,
		help="How many denoising steps for generation.",
	)
	parser.add_argument(
		"--guidance", type=float, default=3,
		help="Guidance scale for classifier free guidance."
	)
	parser.add_argument(
		"--batch_size", type=int, default=8,
		help="Batch size for generation.",
	)
	parser.add_argument(
		"--num_samples", type=int, default=1,
		help="How many samples per prompt.",
	)
	parser.add_argument(
		"--num_test_instances", type=int, default=-1,
		help="How many test instances to evaluate.",
	)
	parser.add_argument(
		"--text2musicfeature_mode", type=str, default="predicted",
		help="NN/RuleBased/GroundTruth/predicted",
	)
	parser.add_argument(
		"--beats_key", type=str, default="beats",
		help="beats",
	)
	parser.add_argument(
		"--chords_key", type=str, default="chords",
		help="chords",
	)
	parser.add_argument(
		"--chords_time_key", type=str, default="chords_time",
		help="chords_time",
	)
	parser.add_argument(
		"--aux_name", type=str, default="out",
		help="auxiliary directory name",
	)
	parser.add_argument(
		"--evaluate", type=bool, default=True,
		help="Evaluate or not",
	)
	args = parser.parse_args()

	return args

class text2music_feature_generator():
	def __init__(self, mode, beat_generator = None, chord_generator = None, max_dur = 10):
		self.mode = mode #"NN"/"Rule based" if rule based generate based on prompts
		self.max_dur = 10
		self.beat_generator = beat_generator
		self.chord_generator = chord_generator
	def extract_beat_number(self, input_string):
		# Regular expression pattern to find the number after "beat"
		pattern = r'beat.*?(\d+)'
		
		# Search for the pattern in the input string
		match = re.search(pattern, input_string, re.IGNORECASE)
		
		if match:
			# Extract the number from the matched group
			beat_number = int(match.group(1))
			return beat_number
		else:
			# No match found
			return None
	def extract_bpm(self, input_string):
		# Regular expression pattern to find the bpm value
		pattern = r'bpm\D*(\d+(?:\.\d+)?)'
		
		# Search for the pattern in the input string
		match = re.search(pattern, input_string, re.IGNORECASE)
		
		if match:
			# Extract the bpm value from the matched group
			bpm = float(match.group(1))
			return bpm
		else:
			# No match found
			return None
	def extract_tempo(self, input_string):
		tempo_mappings = {
			'Grave': 40,
			'Largo': 60,
			'Adagio': 70,
			'Andante': 90,
			'Moderato': 110,
			'Allegro': 140,
			'Vivace': 160,
			'Presto': 210,
			'Prestissimo': 210
		}

		for tempo, bpm in tempo_mappings.items():
			if tempo.lower() in input_string.lower():
				return bpm
		
		# No match found
		return None
	def extract_chords(self, chord_string):
		# Regular expression pattern to find chords with extensions, alterations, and inversions
		pattern = r'([A-Ga-g][b#]?(?:maj7?|m(?:6|7b5)?|7|aug|dim)?(?:/[A-Ga-g][b#])?)'

		# Find all matches of chords in the chord string
		chords = re.findall(pattern, chord_string, re.IGNORECASE)
		
		return chords
	def get_rule_based_beat(self, text, default_beat_num = 4, default_bpm = 70, default_duration = 10):
		extracted_beat_number = self.extract_beat_number(text)
		extracted_bpm = self.extract_bpm(text)
		extracted_bpm_from_tempo = self.extract_tempo(text)

		if extracted_beat_number:
			final_beat_number = extracted_beat_number
		else:
			final_beat_number = default_beat_num
		
		if extracted_bpm:
			final_bpm = extracted_bpm
		elif extracted_tempo:
			final_bpm = extracted_bpm_from_tempo
		else:
			final_bpm = default_bpm

		interval = 60.0 / final_bpm  # Time interval between beats in seconds

		regular_beats = []
		beat_timings = []

		# Generate the regular beats and beat timings
		for i in range(beats):
			regular_beats.append(i % beats + 1)
			beat_timings.append(i * interval)

			if beat_timings[-1] >= self.max_dur:
				break

		return [beat_timings, regular_beats]

	def get_rule_based_chord(self, text, beats_output, default_chord = ["N"], default_duration = ["N"]):
		extracted_chords = self.extract_chord(text)
		if extracted_chords:
			#assign chord to the assigned downbeats #TODO
			#calculate how many downbeats
			num_of_downbeats = beats_output[1].count(1.)
			indices_of_downbeats =  [time for time, tpe in zip(beats_output[0], beats_output[1]) if tpe == 1.]
			assign_everyother = int(num_of_downbeats/len(extracted_chords))
			chord_timing = indices_of_downbeats[::assign_everyother]
			return chord_timing,extracted_chords
		else:
			return default_duration, default_chord

	def generate_beats_chords(self, text_prompt):
		#output format [[0.24, 0.68, 1.08, 1.52, 1.92, 2.32, 2.72, 3.1, 3.5, 3.9, 4.32, 4.72, 5.1, 5.48, 5.86, 6.22, 6.62, 6.98, 7.38, 7.76, 8.12, 8.5, 8.9, 9.28, 9.64], [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]]
		#output format "chords": ["D"] "chords_time": [0.464399092]
		if self.mode == "Rule Based":
			beats_output = self.get_rule_based_beat(text_prompt)
			chord_timing_output, chord_output = self.get_rule_based_chord(text_prompt, beats_output)
		elif self.mode == "NN":
			beats_output = self.beat_generator(text_prompt)
			chord_timing_output, chord_output = self.chord_generator(text_prompt)
		return beats_output, chord_timing_output, chord_output



def main():
	args = parse_args()
	
	train_args = dotdict(json.loads(open(args.original_args).readlines()[0]))

	print("train args:", train_args)
	
	if "hf_model" not in train_args:
		train_args["hf_model"] = None
	
	# Load Models #
	if train_args.hf_model:
		tango = Tango(train_args.hf_model, "cpu")
		vae, stft, model = tango.vae.cuda(), tango.stft.cuda(), tango.model.cuda()
	else:
		name = "audioldm-s-full"
		vae, stft = build_pretrained_models(name)
		vae, stft = vae.cuda(), stft.cuda() #Nic, uncomment when GPU avail
		model = MusicAudioDiffusion(
			train_args.text_encoder_name, train_args.scheduler_name, train_args.unet_model_name, train_args.unet_model_config, train_args.snr_gamma, train_args.freeze_text_encoder, train_args.uncondition
		).cuda()
		model.eval() #Nic, uncomment when GPU avail
	
	# Load Trained Weight #
	device = vae.device()
	model.load_state_dict(torch.load(args.model))
	
	scheduler = DDPMScheduler.from_pretrained(train_args.scheduler_name, subfolder="scheduler")
	evaluator = EvaluationHelper(16000, "cuda:0")
	
	if args.num_samples > 1:
		clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
		clap.eval()
		clap_processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
	
	wandb.init(project="Text to Audio Diffusion Evaluation")

	def audio_text_matching(waveforms, text, sample_freq=16000, max_len_in_seconds=10):
		new_freq = 48000
		resampled = []
		
		for wav in waveforms:
			x = torchaudio.functional.resample(torch.tensor(wav, dtype=torch.float).reshape(1, -1), orig_freq=sample_freq, new_freq=new_freq)[0].numpy()
			resampled.append(x[:new_freq*max_len_in_seconds])

		inputs = clap_processor(text=text, audios=resampled, return_tensors="pt", padding=True, sampling_rate=48000)
		inputs = {k: v.to(device) for k, v in inputs.items()}

		with torch.no_grad():
			outputs = clap(**inputs)

		logits_per_audio = outputs.logits_per_audio
		ranks = torch.argsort(logits_per_audio.flatten(), descending=True).cpu().numpy()
		return ranks
	
	# Load Data #
	if train_args.prefix:
		prefix = train_args.prefix
	else:
		prefix = ""
		
	text_prompts = [json.loads(line)[args.text_key] for line in open(args.test_file).readlines()]
	text_prompts = [prefix + inp for inp in text_prompts]
	if args.text2musicfeature_mode == "predicted":
		args.beats_key="beats_predicted"
		args.chords_key="chords_predicted"
		args.chords_time_key="chords_predicted_time"
	beats_gt = [json.loads(line)[args.beats_key] for line in open(args.test_file).readlines()]
	chords_gt = [json.loads(line)[args.chords_key] for line in open(args.test_file).readlines()]
	chords_timing_gt = [json.loads(line)[args.chords_time_key] for line in open(args.test_file).readlines()]

	
	if args.num_test_instances != - 1:
		text_prompts = text_prompts[:args.num_test_instances]

	#Decide whether to use ground truth music feature or generate based on rules or generate using neural network
	if args.text2musicfeature_mode =="RuleBased":
		#TODO Nic
		music_feature_generator = text2music_feature_generator(mode = args.text2musicfeature_mode)
	elif args.text2musicfeature_mode =="GroundTruth": # USING GROUND TRUTH FEATURES FROM ORIG DATA
		beats = beats_gt
		chords = chords_gt
		chords_timing = chords_timing_gt

	elif args.text2musicfeature_mode =="NN": #TODO later - combine with the predictors
		beat_generator = None
		chord_generator = None #TODO TEXT to music features
		music_feature_generator = text2music_feature_generator(mode = args.text2musicfeature_mode, beat_generator = beat_generator, chord_generator = chord_generator)
	elif args.text2musicfeature_mode == "predicted": # IF FEATURES ARE PREDICTED from the predictor models and retrieved from a file
		beats=[]
		for beat_line in beats_gt:
			beat_timings=beat_line[0]
			beat_count=beat_line[1]
			if len(beat_timings)==0:
				beats.append([[],[]])
				continue
			beat_counts=[]
			if len(beat_timings)>50:
				beat_timings=beat_timings[:50]
			for i in range(len(beat_timings)):
				beat_counts.append(float(1.0+np.mod(i,beat_count)))
			beats.append([beat_timings,beat_counts])
		# beats = beats_gt
		chords = chords_gt
		chords_timing = chords_timing_gt


	# Generate #
	num_steps, guidance, batch_size, num_samples = args.num_steps, args.guidance, args.batch_size, args.num_samples
	all_outputs = []
		
	#TODO integrate music features during inference

	for k in tqdm(range(0, len(text_prompts), batch_size)):
		text = text_prompts[k: k+batch_size]
		beat = beats[k: k+batch_size]
		chord = chords[k: k+batch_size]
		chord_timing = chords_timing[k: k+batch_size]
		
		with torch.no_grad():
			latents = model.inference(text, beat, chord, chord_timing, scheduler, num_steps, guidance, num_samples, disable_progress=True) #TODO, nic change
			mel = vae.decode_first_stage(latents)
			wave = vae.decode_to_waveform(mel)
			all_outputs += [item for item in wave]
			
	# Save #
	exp_id = str(int(time.time()))
	if not os.path.exists("outputs"):
		os.makedirs("outputs")
	
	if num_samples == 1:
		output_dir = "outputs/{}_{}_{}_steps_{}_guidance_{}".format(args.aux_name, exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance)
		os.makedirs(output_dir, exist_ok=True)
		for j, wav in enumerate(all_outputs):
			sf.write("{}/output_{}.wav".format(output_dir, j), wav, samplerate=16000)
		if args.evaluate:
			result = evaluator.main(output_dir, args.test_references)
			result["Steps"] = num_steps
			result["Guidance Scale"] = guidance
			result["Test Instances"] = len(text_prompts)
			wandb.log(result)
			
			result["scheduler_config"] = dict(scheduler.config)
			result["args"] = dict(vars(args))
			result["output_dir"] = output_dir

			with open("outputs/summary.jsonl", "a") as f:
				f.write(json.dumps(result) + "\n\n")
			
	else:
		for i in range(num_samples):
			output_dir = "outputs/{}_{}_{}_steps_{}_guidance_{}/rank_{}".format(args.aux_name, exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, i+1)
			os.makedirs(output_dir, exist_ok=True)
		
		groups = list(chunks(all_outputs, num_samples))
		for k in tqdm(range(len(groups))):
			wavs_for_text = groups[k]
			rank = audio_text_matching(wavs_for_text, text_prompts[k])
			ranked_wavs_for_text = [wavs_for_text[r] for r in rank]
			
			for i, wav in enumerate(ranked_wavs_for_text):
				output_dir = "outputs/{}_{}_{}_steps_{}_guidance_{}/rank_{}".format(args.aux_name, exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, i+1)
				sf.write("{}/output_{}.wav".format(output_dir, k), wav, samplerate=16000)

		if args.evaluate:
			# Compute results for each rank #
			for i in range(num_samples):
				output_dir = "outputs/{}_{}_{}_steps_{}_guidance_{}/rank_{}".format(args.aux_name, exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, i+1)
				result = evaluator.main(output_dir, args.test_references)
				result["Steps"] = num_steps
				result["Guidance Scale"] = guidance
				result["Instances"] = len(text_prompts)
				result["clap_rank"] = i+1
				
				wb_result = copy.deepcopy(result)
				wb_result = {"{}_rank{}".format(k, i+1): v for k, v in wb_result.items()}
				wandb.log(wb_result)
				
				result["scheduler_config"] = dict(scheduler.config)
				result["args"] = dict(vars(args))
				result["output_dir"] = output_dir

				with open("outputs/summary.jsonl", "a") as f:
					f.write(json.dumps(result) + "\n\n")
		
if __name__ == "__main__":
	main()