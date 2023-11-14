import time
import argparse
import json
import logging
import math
import os
import gc
# from tqdm import tqdm
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import wandb
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import soundfile as sf
import diffusers
import transformers
import tools.torch_tools as torch_tools
from huggingface_hub import snapshot_download
from models_speed_up import build_pretrained_models, MusicAudioDiffusion
from transformers import SchedulerType, get_scheduler

from spacy.lang.en import English
import random
from transformers import CLIPTokenizer, AutoTokenizer
from transformers import CLIPTextModel, T5EncoderModel, AutoModel
from layers.layers import chord_tokenizer, beat_tokenizer
# from transformers.optimization import Adafactor
# import bitsandbytes as bnb

import h5py
logger = get_logger(__name__)


def parse_args():
	parser = argparse.ArgumentParser(description="Finetune a diffusion model for text to audio generation task.")
	parser.add_argument(
		"--train_file", type=str, default="data/100train_combined_chatgpt_aug4.json",
		help="A csv or a json file containing the training data."
	)
	parser.add_argument(
		"--validation_file", type=str, default="data/100eval_combined_chatgpt_aug4.json",
		help="A csv or a json file containing the validation data."
	)
	parser.add_argument(
		"--validation_file2", type=str, default="data/100eval_musiccaps_yolo.json",
		help="A csv or a json file containing the validation data."
	)
	parser.add_argument(
		"--test_file", type=str, default="data/100test_musiccaps_ep4.json",
		help="A csv or a json file containing the test data for generation."
	)
	parser.add_argument(
		"--num_examples", type=int, default=-1,
		help="How many examples to use for training and validation.",
	)
	parser.add_argument(
		"--text_encoder_name", type=str, default="google/flan-t5-large",
		help="Text encoder identifier from huggingface.co/models.",
	)
	parser.add_argument(
		"--scheduler_name", type=str, default="stabilityai/stable-diffusion-2-1",
		help="Scheduler identifier.",
	)
	parser.add_argument(
		"--unet_model_name", type=str, default=None,
		help="UNet model identifier from huggingface.co/models.",
	)
	parser.add_argument(
		"--unet_model_config", type=str, default=None,
		help="UNet model config json path.",
	)
	parser.add_argument(
		"--hf_model", type=str, default=None,
		help="Tango model identifier from huggingface: declare-lab/tango",
	)
	parser.add_argument(
		"--snr_gamma", type=float, default=None,
		help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
		"More details here: https://arxiv.org/abs/2303.09556.",
	)
	parser.add_argument(
		"--freeze_text_encoder", action="store_true", default=False,
		help="Freeze the text encoder model.",
	)
	parser.add_argument(
		"--text_column", type=str, default="captions",
		help="The name of the column in the datasets containing the input texts.",
	)
	parser.add_argument(
		"--audio_column", type=str, default="location",
		help="The name of the column in the datasets containing the audio paths.",
	)
	parser.add_argument(
		"--beats_column", type=str, default="beats",
		help="The name of the column in the datasets containing the beats music feature.",
	)
	parser.add_argument(
		"--beat_len", type=str, default=50,
		help="len of beat feat.",
	)
	parser.add_argument(
		"--chord_len", type=str, default=20,
		help="len of chord feat.",
	)
	parser.add_argument(
		"--text_max_len", type=str, default=183,
		help="len of text_prompt",
	)
	parser.add_argument(
		"--chords_column", type=str, default="chords",
		help="The name of the column in the datasets containing the chords music feature.",
	)
	parser.add_argument(
		"--chords_time_column", type=str, default="chords_time",
		help="The name of the column in the datasets containing the chords music feature.",
	)
	parser.add_argument(
		"--augment", action="store_true", default=False,
		help="Augment training data.",
	)
	parser.add_argument(
		"--uncondition", action="store_true", default=False,
		help="10% uncondition for training.",
	)
	parser.add_argument(
		"--drop_sentences", action="store_true", default=False,
		help="Allow preset sentence dropping when loading the data.",
	)
	parser.add_argument(
		"--prefix", type=str, default=None,
		help="Add prefix in text prompts.",
	)
	parser.add_argument(
		"--per_device_train_batch_size", type=int, default=1,
		help="Batch size (per device) for the training dataloader.",
	)
	parser.add_argument(
		"--per_device_eval_batch_size", type=int, default=1,
		help="Batch size (per device) for the validation dataloader.",
	)
	parser.add_argument(
		"--learning_rate", type=float, default=4.5e-5,
		help="Initial learning rate (after the potential warmup period) to use.",
	)
	parser.add_argument(
		"--weight_decay", type=float, default=1e-8,
		help="Weight decay to use."
	)
	parser.add_argument(
		"--num_train_epochs", type=int, default=40,
		help="Total number of training epochs to perform."
	)
	parser.add_argument(
		"--max_train_steps", type=int, default=None,
		help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
	)
	parser.add_argument(
		"--gradient_accumulation_steps", type=int, default=8,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument(
		"--lr_scheduler_type", type=SchedulerType, default="linear",
		help="The scheduler type to use.",
		choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
	)
	parser.add_argument(
		"--num_warmup_steps", type=int, default=0,
		help="Number of steps for the warmup in the lr scheduler."
	)
	parser.add_argument(
		"--adam_beta1", type=float, default=0.9,
		help="The beta1 parameter for the Adam optimizer."
	)
	parser.add_argument(
		"--adam_beta2", type=float, default=0.999,
		help="The beta2 parameter for the Adam optimizer."
	)
	parser.add_argument(
		"--adam_weight_decay", type=float, default=1e-2,
		help="Weight decay to use."
	)
	parser.add_argument(
		"--adam_epsilon", type=float, default=1e-08,
		help="Epsilon value for the Adam optimizer"
	)
	parser.add_argument(
		"--output_dir", type=str, default=None,
		help="Where to store the final model."
	)
	parser.add_argument(
		"--seed", type=int, default=None,
		help="A seed for reproducible training."
	)
	parser.add_argument(
		"--checkpointing_steps", type=str, default="best",
		help="Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases.",
	)
	parser.add_argument(
		"--save_every", type=int, default=1,
		help="Save model after every how many epochs when checkpointing_steps is set to best."
	)
	parser.add_argument(
		"--resume_from_checkpoint", type=str, default=None,
		help="If the training should continue from a local checkpoint folder.",
	)
	parser.add_argument(
		"--with_tracking", action="store_true",
		help="Whether to enable experiment trackers for logging.",
	)
	parser.add_argument(
		"--report_to", type=str, default="all",
		help=(
			'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
			' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
			"Only applicable when `--with_tracking` is passed."
		),
	)
	args = parser.parse_args()

	# Sanity checks
	if args.train_file is None and args.validation_file is None:
		raise ValueError("Need a training/validation file.")
	else:
		if args.train_file is not None:
			extension = args.train_file.split(".")[-1]
			assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
		if args.validation_file is not None:
			extension = args.validation_file.split(".")[-1]
			assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

	return args



	
class Text2AudioDataset_Speedup(Dataset):
	def __init__(self, dataset, prefix, text_column, audio_column, beats_column, chords_column, chords_time_column, num_examples=-1, pretrained_audioldm = None, pretrained_stft = None, target_length = 1024, pretrained_text_tokenizer = None, pretrained_text_encoder = None,beat_len = None, chord_len = None, text_max_len = 183, preprocessed_audio_latent_path =None, preprocessed_text_embedding_path = None, preprocessed_text_mask_path = None):
		##build pretrained model
		self.pretrained_audioldm = pretrained_audioldm
		self.pretrained_stft = pretrained_stft
		self.tokenizer = pretrained_text_tokenizer
		self.text_encoder = pretrained_text_encoder
		self.beat_tokenizer = beat_tokenizer(seq_len_beat=beat_len, if_pad = True)
		self.chord_tokenizer = chord_tokenizer(seq_len_chord=chord_len, if_pad = True)
		self.target_length = target_length
		self.text_max_len = text_max_len 
		
		#load preprocessed file/encode using pretrained model
		if os.path.isfile(preprocessed_audio_latent_path): #TODO: ADD WARNING, IF DATA PATH CHANGES, MUST REMOVE THE PREPROCESSED DATA FIRST S.T. THE ORDER OF DATA IS PRESERVED
			# self.audio_latent = torch.load(preprocessed_audio_latent_path)
			self.audio_latent = h5py.File(preprocessed_audio_latent_path, 'r')['audio_latent']
			
		else:
			self.audios = list(dataset[audio_column])
			self.audio_latent = self.encode_audio(self.audios)
			# torch.save(self.audio_latent, preprocessed_audio_latent_path)
		
			audio_latent_f = h5py.File(preprocessed_audio_latent_path, 'w')
			audio_latent_f.create_dataset("audio_latent", data=self.audio_latent.numpy())
			audio_latent_f.close()

		
		
		
		
		if os.path.isfile(preprocessed_text_embedding_path) and os.path.isfile(preprocessed_text_mask_path):
			# self.text_mask = torch.load(preprocessed_text_mask_path)
			# print("fininished loading mask ")

			# if preprocessed_text_embedding_path.split(".")[-1]=="pt":
			# 	self.text_latent = torch.load(preprocessed_text_embedding_path)
			# 	print("finished pt!")
			# elif preprocessed_text_embedding_path.split(".")[-1]=="h5":
			# 	# self.text_latent = torch.load(preprocessed_text_embedding_path)
			self.text_latent = h5py.File(preprocessed_text_embedding_path, 'r')['text_embedding_train']
			self.text_mask = h5py.File(preprocessed_text_mask_path, 'r')['text_mask_train']
			print("fininisheD H5 ")
		
		
		else:
			inputs = list(dataset[text_column])
			self.inputs = [prefix + inp for inp in inputs]
			self.text_latent, self.text_mask = self.encode_text(self.inputs) 
			
			text_latent = h5py.File(preprocessed_text_embedding_path, 'w')
			text_latent.create_dataset("text_embedding_train", data=self.text_latent.numpy())
			text_latent.close()

			text_mask = h5py.File(preprocessed_text_mask_path, 'w')
			text_mask.create_dataset("text_mask_train", data=self.text_mask.numpy())
			text_mask.close()

			# Close the HDF5 file when done
			
			# torch.save(self.text_latent, preprocessed_text_embedding_path)
			# torch.save(self.text_mask, preprocessed_text_mask_path)
		
		
		self.beats = list(dataset[beats_column]) 
		self.beat_tokenized,self.beat_timing, self.beat_mask = self.tokenize_beats(self.beats)

		self.chords = list(dataset[chords_column])
		self.chords_time = list(dataset[chords_time_column])
		self.chord_root_tokenized, self.chord_type_tokenized, self.chord_inv_tokenized, self.chord_timing, self.chord_mask = self.tokenize_chords(self.chords, self.chords_time)

	def encode_audio(self, audios, batch_size = 16):
		out = []
		batched_list = [audios[i:i+batch_size] for i in range(0, len(audios), batch_size)]


		# batched_list = batched_list[int(len(batched_list)*0.75):]


		for i, batch_audio in enumerate(batched_list):
			print("progress", batch_audio,i, "/", len(batched_list), i/len(batched_list))
			mel, _, _ = torch_tools.wav_to_fbank(batch_audio, self.target_length, self.pretrained_stft)
			mel = mel.unsqueeze(1).cuda()
			first_stage = self.pretrained_audioldm.encode_first_stage(mel)
			true_latent = self.pretrained_audioldm.get_first_stage_encoding(first_stage) #batch, channels, time_compressed, freq_compressed [2, 8, 256, 16]
			true_latent = true_latent.detach().cpu()
			out.append(true_latent)

		return torch.cat(out) 
	
	def encode_text(self, inputs, batch_size = 16):
		out_encoded, out_mask = [], []
		# max_len = 0
		batched_list = [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]
		for i, text in enumerate(batched_list):
			batch = self.tokenizer(
				text, max_length=self.text_max_len, padding="max_length", truncation=True, return_tensors="pt"
			)
			# batch = self.tokenizer(
			# 	text, max_length=self.text_max_len, padding=True, truncation=True, return_tensors="pt"
			# )

			input_ids, attention_mask = batch.input_ids.cuda(), batch.attention_mask.cuda() #cuda
			# if len(input_ids[0])>max_len:
			# 	max_len = len(input_ids[0])
			with torch.no_grad():
				encoder_hidden_states = self.text_encoder(
					input_ids=input_ids, attention_mask=attention_mask
				)[0] #batch, len_text, dim
			boolean_encoder_mask = (attention_mask == 1) #https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py#L687C1-L693C23
			print("encoding text process", i*batch_size, "/", len(inputs), i*batch_size/len(inputs))

			# print("checking text shape", boolean_encoder_mask.shape, encoder_hidden_states.shape)
			
			out_encoded.append(encoder_hidden_states.detach().cpu())
			out_mask.append(boolean_encoder_mask.detach().cpu())
		# print("max len", max_len)
		return torch.cat(out_encoded), torch.cat(out_mask) 
	
	def tokenize_beats(self, beats): 
		# device = self.beat_embedding_layer.device
		out_beat = []
		out_beat_timing = []
		out_mask = []
		for beat in beats:
			tokenized_beats,tokenized_beats_timing, tokenized_beat_mask = self.beat_tokenizer(beat)
			out_beat.append(tokenized_beats)
			out_beat_timing.append(tokenized_beats_timing)
			out_mask.append(tokenized_beat_mask)
		# out_beat, out_beat_timing, out_mask = torch.tensor(out_beat).to(device), torch.tensor(out_beat_timing).to(device), torch.tensor(out_mask).to(device) #batch, len_beat
		out_beat, out_beat_timing, out_mask = torch.tensor(out_beat), torch.tensor(out_beat_timing), torch.tensor(out_mask) #batch, len_beat
		# embedded_beat = self.beat_embedding_layer(out_beat, out_beat_timing)

		return out_beat, out_beat_timing, out_mask

	def tokenize_chords(self, chords,chords_time): 
		
		out_chord_root = []
		out_chord_type = []
		out_chord_inv = []
		out_chord_timing = []
		out_mask = []
		for chord, chord_time in zip(chords,chords_time): #batch loop
			tokenized_chord_root, tokenized_chord_type, tokenized_chord_inv, tokenized_chord_time, tokenized_chord_mask = self.chord_tokenizer(chord, chord_time)
			out_chord_root.append(tokenized_chord_root)
			out_chord_type.append(tokenized_chord_type)
			out_chord_inv.append(tokenized_chord_inv)
			out_chord_timing.append(tokenized_chord_time)
			out_mask.append(tokenized_chord_mask)
		#chords: (B, LEN, 4)
		out_chord_root, out_chord_type, out_chord_inv, out_chord_timing, out_mask = torch.tensor(out_chord_root), torch.tensor(out_chord_type), torch.tensor(out_chord_inv), torch.tensor(out_chord_timing), torch.tensor(out_mask)
		return out_chord_root, out_chord_type, out_chord_inv, out_chord_timing, out_mask

	def __len__(self):
		return len(self.audio_latent)

	# def get_num_instances(self):
	# 	return len(self.inputs)

	def __getitem__(self, index):
		text_embedded, text_mask, audio_embedded, beat_tokenized, beat_timing, beat_mask, chord_root_tokenized, chord_type_tokenized, chord_inv_tokenized, chord_timing, chord_mask = self.text_latent[index], self.text_mask[index], self.audio_latent[index], self.beat_tokenized[index],self.beat_timing[index], self.beat_mask[index], self.chord_root_tokenized[index], self.chord_type_tokenized[index], self.chord_inv_tokenized[index], self.chord_timing[index], self.chord_mask[index]
		return text_embedded, text_mask, audio_embedded, beat_tokenized, beat_timing, beat_mask, chord_root_tokenized, chord_type_tokenized, chord_inv_tokenized, chord_timing, chord_mask
	
	# def collate_fn(self, data):
	# 	dat = pd.DataFrame(data)
	# 	return [dat[i].tolist() for i in dat]
   

def main():
	args = parse_args()
	accelerator_log_kwargs = {}

	if args.with_tracking:
		accelerator_log_kwargs["log_with"] = args.report_to
		accelerator_log_kwargs["logging_dir"] = args.output_dir

	accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
	
	# Make one log on every process with the configuration for debugging.
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO,
	)
	logger.info(accelerator.state, main_process_only=False)

	datasets.utils.logging.set_verbosity_error()
	diffusers.utils.logging.set_verbosity_error()
	transformers.utils.logging.set_verbosity_error()

	# If passed along, set the training seed now.
	if args.seed is not None:
		set_seed(args.seed)

	# Handle output directory creation and wandb tracking
	if accelerator.is_main_process:
		if args.output_dir is None or args.output_dir == "":
			args.output_dir = "saved/" + str(int(time.time()))
			
			if not os.path.exists("saved"):
				os.makedirs("saved")
				
			os.makedirs(args.output_dir, exist_ok=True)
			
		elif args.output_dir is not None:
			os.makedirs(args.output_dir, exist_ok=True)

		os.makedirs("{}/{}".format(args.output_dir, "outputs"), exist_ok=True)
		with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
			f.write(json.dumps(dict(vars(args))) + "\n\n")

		accelerator.project_configuration.automatic_checkpoint_naming = False

		wandb.init(project="Text to Audio Diffusion")

	accelerator.wait_for_everyone()

	# Get the datasets
	data_files = {}
	if args.train_file is not None:
		data_files["train"] = args.train_file
	if args.validation_file is not None:
		data_files["validation"] = args.validation_file
		data_files["validation2"] = args.validation_file2#NEW

	if args.test_file is not None:
		data_files["test"] = args.test_file
	else:
		if args.validation_file is not None:
			data_files["test"] = args.validation_file

	extension = args.train_file.split(".")[-1]
	raw_datasets = load_dataset(extension, data_files=data_files)
	text_column, audio_column, beats_column, chords_column, chords_time_column = args.text_column, args.audio_column, args.beats_column, args.chords_column, args.chords_time_column #upd Nic delete
	
	



	model = MusicAudioDiffusion(
		args.text_encoder_name, args.scheduler_name, args.unet_model_name, args.unet_model_config, args.snr_gamma, args.freeze_text_encoder, args.uncondition
	)

	if args.hf_model:
		hf_model_path = snapshot_download(repo_id=args.hf_model)
		model.load_state_dict(torch.load("{}/pytorch_model_main.bin".format(hf_model_path), map_location="cpu"))
		accelerator.print("Successfully loaded checkpoint from:", args.hf_model)
		
	if args.prefix:
		prefix = args.prefix
	else:
		prefix = ""
	




	# Initialize pretrained models
	pretrained_model_name = "audioldm-s-full"
	vae, stft = build_pretrained_models(pretrained_model_name)
	vae.eval()
	stft.eval()

	if "stable-diffusion" in args.text_encoder_name:
		pretrained_text_tokenizer = CLIPTokenizer.from_pretrained(args.text_encoder_name, subfolder="tokenizer")
		pretrained_text_encoder = CLIPTextModel.from_pretrained(args.text_encoder_name, subfolder="text_encoder").eval().cuda()
	elif "t5" in args.text_encoder_name:
		pretrained_text_tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name)
		pretrained_text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_name).eval().cuda()
	else:
		pretrained_text_tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name)
		pretrained_text_encoder = AutoModel.from_pretrained(args.text_encoder_name).eval().cuda()
	with accelerator.main_process_first():
		test_dataset = Text2AudioDataset_Speedup(raw_datasets["test"], prefix, text_column, audio_column, beats_column, chords_column, chords_time_column, args.num_examples, pretrained_audioldm= vae.cuda(), pretrained_stft= stft, pretrained_text_tokenizer = pretrained_text_tokenizer, pretrained_text_encoder = pretrained_text_encoder, beat_len = args.beat_len, chord_len = args.chord_len, text_max_len = args.text_max_len, preprocessed_audio_latent_path="preprocessed_input_data/test_audio_latent.h5", preprocessed_text_mask_path="preprocessed_input_data/test_text_mask.h5", preprocessed_text_embedding_path="preprocessed_input_data/test_text_embedding.h5")
		eval_dataset = Text2AudioDataset_Speedup(raw_datasets["validation"], prefix, text_column, audio_column, beats_column, chords_column, chords_time_column, args.num_examples, pretrained_audioldm= vae.cuda(), pretrained_stft= stft, pretrained_text_tokenizer = pretrained_text_tokenizer, pretrained_text_encoder = pretrained_text_encoder,  beat_len = args.beat_len, chord_len = args.chord_len, text_max_len = args.text_max_len, preprocessed_audio_latent_path="preprocessed_input_data/eval1_audio_latent.h5", preprocessed_text_mask_path="preprocessed_input_data/eval1_text_mask.h5", preprocessed_text_embedding_path="preprocessed_input_data/eval1_text_embedding.h5")
		eval_dataset2 = Text2AudioDataset_Speedup(raw_datasets["validation2"], prefix, text_column, audio_column, beats_column, chords_column, chords_time_column, args.num_examples, pretrained_audioldm= vae.cuda(), pretrained_stft= stft, pretrained_text_tokenizer = pretrained_text_tokenizer, pretrained_text_encoder = pretrained_text_encoder, beat_len = args.beat_len, chord_len = args.chord_len, text_max_len = args.text_max_len, preprocessed_audio_latent_path="preprocessed_input_data/eval2_audio_latent.h5", preprocessed_text_mask_path="preprocessed_input_data/eval2_text_mask.h5", preprocessed_text_embedding_path="preprocessed_input_data/eval2_text_embedding.h5")
		train_dataset = Text2AudioDataset_Speedup(raw_datasets["train"], prefix, text_column, audio_column, beats_column, chords_column, chords_time_column, args.num_examples, pretrained_audioldm= vae.cuda(), pretrained_stft= stft, pretrained_text_tokenizer = pretrained_text_tokenizer, pretrained_text_encoder = pretrained_text_encoder, beat_len = args.beat_len, chord_len = args.chord_len, text_max_len = args.text_max_len, preprocessed_audio_latent_path="preprocessed_input_data/train_audio_latent.h5", preprocessed_text_mask_path="preprocessed_input_data/train_text_mask.h5", preprocessed_text_embedding_path="preprocessed_input_data/train_text_embedding.h5")
		print("debug, finish train dataset loading")
		accelerator.print("Num instances in train: {}, validation: {}, test: {}".format(len(train_dataset), len(eval_dataset), len(test_dataset)))
	train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size)
	eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size)
	eval_dataloader2 = DataLoader(eval_dataset2, shuffle=False, batch_size=args.per_device_eval_batch_size)
	test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size)
	print("debug, finish data loading")
	#remove pretrained model from memory #TODO check if removed from memory 
	del vae
	del pretrained_text_encoder

	# Optimizer
	if args.freeze_text_encoder:
		# for param in model.text_encoder.parameters():
		# 	param.requires_grad = False
		# 	model.text_encoder.eval()
		if args.unet_model_config:
			# optimizer_parameters = model.unet.parameters()
			optimizer_parameters = list(model.unet.parameters()) + list(model.beat_embedding_layer.parameters()) + list(model.chord_embedding_layer.parameters())

			accelerator.print("Optimizing UNet parameters.")
		else:
			optimizer_parameters = list(model.unet.parameters()) + list(model.group_in.parameters()) + list(model.group_out.parameters())
			accelerator.print("Optimizing UNet and channel transformer parameters.")
	else:
		optimizer_parameters = model.parameters()
		accelerator.print("Optimizing Text Encoder and UNet parameters.")

	num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
	accelerator.print("Num trainable parameters: {}".format(num_trainable_parameters))

	optimizer = torch.optim.AdamW(
		optimizer_parameters, lr=args.learning_rate,
		betas=(args.adam_beta1, args.adam_beta2),
		weight_decay=args.adam_weight_decay,
		eps=args.adam_epsilon,
	)

	print("debug, finish prep optimizer")

	# vae, stft, model = accelerator.prepare(
	#     vae, stft, model
	# )

	# optimizer = bnb.optim.Adam8bit(optimizer_parameters, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2)) # add bnb optimizer
	# optimizer = Adafactor(optimizer_parameters, lr=args.learning_rate, scale_parameter=False, relative_step=False)

	# Scheduler and math around the number of training steps.
	overrode_max_train_steps = False
	num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
	if args.max_train_steps is None:
		args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
		overrode_max_train_steps = True

	lr_scheduler = get_scheduler(
		name=args.lr_scheduler_type,
		optimizer=optimizer,
		num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
		num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
	)

	# optimizer, lr_scheduler  = accelerator.prepare(
		# optimizer, lr_scheduler
	# )
	# Prepare everything with our `accelerator`.
	# vae, stft, model, optimizer, lr_scheduler = accelerator.prepare(
	# 	vae, stft, model, optimizer, lr_scheduler
	# )
	model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
	# print("checking model", model)
	# print("check device", model.parameters())
	# for name, param in model.named_parameters():
	#     if not param.requires_grad:
	#         print(name, param.data.device)

	train_dataloader, eval_dataloader, eval_dataloader2, test_dataloader = accelerator.prepare(train_dataloader, eval_dataloader, eval_dataloader2, test_dataloader)
	print("debug, finish prep train dataloader")
	# We need to recalculate our total training steps as the size of the training dataloader may have changed.
	num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
	if overrode_max_train_steps:
		args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
	# Afterwards we recalculate our number of training epochs
	args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

	# Figure out how many steps we should save the Accelerator states
	checkpointing_steps = args.checkpointing_steps
	if checkpointing_steps is not None and checkpointing_steps.isdigit():
		checkpointing_steps = int(checkpointing_steps)

	# We need to initialize the trackers we use, and also store our configuration.
	# The trackers initializes automatically on the main process.
	if args.with_tracking:
		experiment_config = vars(args)
		# TensorBoard cannot log Enums, need the raw value
		experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
		accelerator.init_trackers("text_to_audio_diffusion", experiment_config)

	# Train!
	total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

	logger.info("***** Running training *****")
	logger.info(f"  Num examples = {len(train_dataset)}")
	logger.info(f"  Num Epochs = {args.num_train_epochs}")
	logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
	logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
	logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
	logger.info(f"  Total optimization steps = {args.max_train_steps}")

	# Only show the progress bar once on each machine.
	progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

	completed_steps = 0
	starting_epoch = 0
	# Potentially load in the weights and states from a previous save
	if args.resume_from_checkpoint:
		if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
			accelerator.load_state(args.resume_from_checkpoint)
			# path = os.path.basename(args.resume_from_checkpoint)
			accelerator.print(f"Resumed from local checkpoint: {args.resume_from_checkpoint}")
		else:
			# Get the most recent checkpoint
			dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
			dirs.sort(key=os.path.getctime)
			# path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
			
	# Duration of the audio clips in seconds
	duration, best_loss = 10, np.inf
	if args.drop_sentences: #Nic: this is bypassed by default
		print('Drop_sentence is set to True, initializing spacy sentencizer')
		nlp = English()
		nlp.add_pipe("sentencizer")
		sent_lengths=[]
		for step, batch in enumerate(train_dataloader):
			text, audios, beats, chords,chords_time, _ = batch
			for i in range(len(text)):
				sent_lengths.append(len(list(nlp(text[i]).sents)))
		# sent_length_mean=4.5
		sent_length_mean=np.mean(np.array(sent_lengths))


	for epoch in range(starting_epoch, args.num_train_epochs):
		model.train()
		total_loss, total_val_loss, total_val_loss2 = 0, 0, 0
		for i, batch in enumerate(train_dataloader):
			# print(f"epoch:{epoch}, process: {i/len(train_dataloader)}")
			with accelerator.accumulate(model):
				text_embedded, text_mask, audio_embedded, beat_tokenized, beat_timing, beat_mask, chord_root_tokenized, chord_type_tokenized, chord_inv_tokenized, chord_timing, chord_mask = batch #Nic
				target_length = int(duration * 102.4)
				if args.drop_sentences: #Nic: this is bypassed by default
					text_out=[]
					for i in range(len(text)):
						sentences = list(nlp(text[i]).sents)
						sent_length = len(sentences)
						drop_binary = (random.random()*sent_length/sent_length_mean) < 0.2
						if drop_binary:
							if sent_length<4:
								how_many_to_drop = int(np.floor((20 + random.random()*30)/100*sent_length)) #between 20 and 50 percent of sentences
							else:
								how_many_to_drop = int(np.ceil((20 + random.random()*30)/100*sent_length)) #between 20 and 50 percent of sentences
							which_to_drop = np.random.choice(sent_length,how_many_to_drop,replace=False)
							new_sentences = [sentences[i] for i in range(sent_length) if i not in which_to_drop.tolist()]
							new_sentences = " ".join([new_sentences[i].text for i in range(len(new_sentences))]) #combine sentences back with a space
							text_out.append(new_sentences)
						else:
							text_out.append(text[i])
					text=text_out

				loss = model(text_embedded, text_mask, audio_embedded, beat_tokenized, beat_timing, beat_mask, chord_root_tokenized, chord_type_tokenized, chord_inv_tokenized, chord_timing, chord_mask, validation_mode=False) 
				total_loss += loss.detach().float()
				accelerator.backward(loss)
				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()

			# Checks if the accelerator has performed an optimization step behind the scenes
			if accelerator.sync_gradients:
				progress_bar.update(1)
				completed_steps += 1

			if isinstance(checkpointing_steps, int):
				if completed_steps % checkpointing_steps == 0:
					output_dir = f"step_{completed_steps }"
					if args.output_dir is not None:
						output_dir = os.path.join(args.output_dir, output_dir)
					accelerator.save_state(output_dir)

			if completed_steps >= args.max_train_steps:
				break

		model.eval()
		model.uncondition = False

		eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
		for step, batch in enumerate(eval_dataloader):
			with accelerator.accumulate(model) and torch.no_grad():
				device = model.device
				text_embedded, text_mask, audio_embedded, beat_tokenized, beat_timing, beat_mask, chord_root_tokenized, chord_type_tokenized, chord_inv_tokenized, chord_timing, chord_mask = batch
				target_length = int(duration * 102.4)

				# unwrapped_vae = accelerator.unwrap_model(vae)
				# mel, _, waveform = torch_tools.wav_to_fbank(audios, target_length, stft)
				# mel = mel.unsqueeze(1).to(device)
				# true_latent = unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel))

				val_loss = model(text_embedded, text_mask, audio_embedded, beat_tokenized, beat_timing, beat_mask, chord_root_tokenized, chord_type_tokenized, chord_inv_tokenized, chord_timing, chord_mask, validation_mode=True)
				total_val_loss += val_loss.detach().float()
				eval_progress_bar.update(1)

		eval_progress_bar2 = tqdm(range(len(eval_dataloader2)), disable=not accelerator.is_local_main_process)
		for step, batch in enumerate(eval_dataloader2):
			with accelerator.accumulate(model) and torch.no_grad():
				device = model.device
				text_embedded, text_mask, audio_embedded, beat_tokenized, beat_timing, beat_mask, chord_root_tokenized, chord_type_tokenized, chord_inv_tokenized, chord_timing, chord_mask = batch
				target_length = int(duration * 102.4)

				# unwrapped_vae = accelerator.unwrap_model(vae)
				# mel, _, waveform = torch_tools.wav_to_fbank(audios, target_length, stft)
				# mel = mel.unsqueeze(1).to(device)
				# true_latent = unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel))

				val_loss2 = model(text_embedded, text_mask, audio_embedded, beat_tokenized, beat_timing, beat_mask, chord_root_tokenized, chord_type_tokenized, chord_inv_tokenized, chord_timing, chord_mask, validation_mode=True)
				total_val_loss2 += val_loss2.detach().float()
				eval_progress_bar2.update(1)

		model.uncondition = args.uncondition

		if accelerator.is_main_process:    
			result = {}
			result["epoch"] = epoch+1,
			result["step"] = completed_steps
			result["train_loss"] = round(total_loss.item()/len(train_dataloader), 4)
			result["val_loss"] = round(total_val_loss.item()/len(eval_dataloader), 4)
			result["val_loss2"] = round(total_val_loss2.item()/len(eval_dataloader2), 4)

			wandb.log(result)

			result_string = "Epoch: {}, Loss Train: {}, Val: {}, Val2: {}\n".format(epoch, result["train_loss"], result["val_loss"], result["val_loss2"])
			
			accelerator.print(result_string)

			with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
				f.write(json.dumps(result) + "\n\n")

			logger.info(result)

			if result["val_loss"] < best_loss:
				best_loss = result["val_loss"]
				save_checkpoint = True
			else:
				save_checkpoint = False

		if args.with_tracking:
			accelerator.log(result, step=completed_steps)

		accelerator.wait_for_everyone()
		if accelerator.is_main_process and args.checkpointing_steps == "best":
			if save_checkpoint:
				accelerator.save_state("{}/{}".format(args.output_dir, "best"))
				
			if (epoch + 1) % args.save_every == 0:
				accelerator.save_state("{}/{}".format(args.output_dir, "epoch_" + str(epoch+1)))

		if accelerator.is_main_process and args.checkpointing_steps == "epoch":
			accelerator.save_state("{}/{}".format(args.output_dir, "epoch_" + str(epoch+1)))
			
			
if __name__ == "__main__":
	main()
