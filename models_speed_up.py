import yaml
import random
import inspect
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
from tools.torch_tools import wav_to_fbank

from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from audioldm.utils import default_audioldm_config, get_metadata

from transformers import CLIPTokenizer, AutoTokenizer
from transformers import CLIPTextModel, T5EncoderModel, AutoModel

import diffusers
from diffusers.utils import randn_tensor
from diffusers import DDPMScheduler, UNet2DConditionModel, UNet2DConditionModelMusic
from diffusers import AutoencoderKL as DiffuserAutoencoderKL
from layers.layers import chord_tokenizer, beat_tokenizer, Chord_Embedding, Beat_Embedding, Music_PositionalEncoding, Fundamental_Music_Embedding

def build_pretrained_models(name):
	checkpoint = torch.load(get_metadata()[name]["path"], map_location="cpu")
	scale_factor = checkpoint["state_dict"]["scale_factor"].item()

	vae_state_dict = {k[18:]: v for k, v in checkpoint["state_dict"].items() if "first_stage_model." in k}

	config = default_audioldm_config(name)
	vae_config = config["model"]["params"]["first_stage_config"]["params"]
	vae_config["scale_factor"] = scale_factor

	vae = AutoencoderKL(**vae_config)
	vae.load_state_dict(vae_state_dict)

	fn_STFT = TacotronSTFT(
		config["preprocessing"]["stft"]["filter_length"],
		config["preprocessing"]["stft"]["hop_length"],
		config["preprocessing"]["stft"]["win_length"],
		config["preprocessing"]["mel"]["n_mel_channels"],
		config["preprocessing"]["audio"]["sampling_rate"],
		config["preprocessing"]["mel"]["mel_fmin"],
		config["preprocessing"]["mel"]["mel_fmax"],
	)

	vae.eval()
	fn_STFT.eval()
	return vae, fn_STFT


class MusicAudioDiffusion(nn.Module):
	def __init__(
		self,
		text_encoder_name,
		scheduler_name,
		unet_model_name=None,
		unet_model_config_path=None,
		snr_gamma=None,
		freeze_text_encoder=True,
		uncondition=False,

		d_fme = 1024,  #FME
		fme_type = "se", 
		base = 1, 
		if_trainable = True, 
		translation_bias_type = "nd",
		emb_nn = True,
		d_pe = 1024, #PE
		if_index = True, 
		if_global_timing = True,
		if_modulo_timing = False,
		d_beat = 1024, #Beat
		d_oh_beat_type = 7, 
		beat_len = 50,
		d_chord = 1024, #Chord
		d_oh_chord_type = 12,
		d_oh_inv_type = 4,
		chord_len = 20,

	):
		super().__init__()

		assert unet_model_name is not None or unet_model_config_path is not None, "Either UNet pretrain model name or a config file path is required"

		self.text_encoder_name = text_encoder_name
		self.scheduler_name = scheduler_name
		self.unet_model_name = unet_model_name
		self.unet_model_config_path = unet_model_config_path
		self.snr_gamma = snr_gamma
		self.freeze_text_encoder = freeze_text_encoder
		self.uncondition = uncondition

		# https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
		self.noise_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")
		self.inference_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")

		if unet_model_config_path:
			unet_config = UNet2DConditionModelMusic.load_config(unet_model_config_path)
			self.unet = UNet2DConditionModelMusic.from_config(unet_config, subfolder="unet")
			self.set_from = "random"
			print("UNet initialized randomly.")
		else:
			self.unet = UNet2DConditionModel.from_pretrained(unet_model_name, subfolder="unet")
			self.set_from = "pre-trained"
			self.group_in = nn.Sequential(nn.Linear(8, 512), nn.Linear(512, 4))
			self.group_out = nn.Sequential(nn.Linear(4, 512), nn.Linear(512, 8))
			print("UNet initialized from stable diffusion checkpoint.")

		#Music Feature Encoder
		self.FME = Fundamental_Music_Embedding(d_model = d_fme, base= base, if_trainable = False, type = fme_type,emb_nn=emb_nn,translation_bias_type = translation_bias_type)
		self.PE = Music_PositionalEncoding(d_model = d_pe, if_index = if_index, if_global_timing = if_global_timing, if_modulo_timing = if_modulo_timing, device = "cpu")
		# self.PE2 = Music_PositionalEncoding(d_model = d_pe, if_index = if_index, if_global_timing = if_global_timing, if_modulo_timing = if_modulo_timing, device = self.device)
		self.beat_embedding_layer = Beat_Embedding(self.PE, d_model = d_beat, d_oh_beat_type = d_oh_beat_type)
		self.chord_embedding_layer = Chord_Embedding(self.FME, self.PE, d_model = d_chord, d_oh_type = d_oh_chord_type, d_oh_inv = d_oh_inv_type)


	def compute_snr(self, timesteps):
		"""
		Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
		"""
		alphas_cumprod = self.noise_scheduler.alphas_cumprod
		sqrt_alphas_cumprod = alphas_cumprod**0.5
		sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

		# Expand the tensors.
		# Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
		sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
		while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
			sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
		alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

		sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
		while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
			sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
		sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

		# Compute SNR.
		snr = (alpha / sigma) ** 2
		return snr





	def forward(self, text_embedded, text_mask, audio_embedded, beat_tokenized, beat_timing, beat_mask, chord_root_tokenized, chord_type_tokenized, chord_inv_tokenized, chord_timing, chord_mask, validation_mode=False):
		
		#embed chord and beat
		embedded_beat = self.beat_embedding_layer(beat_tokenized, beat_timing)
		embedded_chord = self.chord_embedding_layer(chord_root_tokenized, chord_type_tokenized, chord_inv_tokenized, chord_timing)
		
		
		device = "cpu"
		num_train_timesteps = self.noise_scheduler.num_train_timesteps
		self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)


		if self.uncondition:
			mask_indices = [k for k in range(len(text_embedded)) if random.random() < 0.1]
			if len(mask_indices) > 0:
				text_embedded[mask_indices] = 0

		if validation_mode:
			timesteps = (self.noise_scheduler.num_train_timesteps//2) * torch.ones((audio_embedded.shape[0],), dtype=torch.int64, device=device)
		else:

			timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (audio_embedded.shape[0],), device=device)
		
		
		timesteps = timesteps.long().cuda()

		noise = torch.randn_like(audio_embedded)
		noisy_latents = self.noise_scheduler.add_noise(audio_embedded, noise, timesteps)

		# Get the target for loss depending on the prediction type
		if self.noise_scheduler.config.prediction_type == "epsilon":
			target = noise
		elif self.noise_scheduler.config.prediction_type == "v_prediction":
			target = self.noise_scheduler.get_velocity(audio_embedded, noise, timesteps)
		else:
			raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

		if self.set_from == "random":
			# model_pred = torch.zeros((bsz,8,256,16)).to(device)
			model_pred = self.unet(
				noisy_latents, timesteps, text_embedded, embedded_beat, embedded_chord,
				encoder_attention_mask=text_mask, beat_attention_mask = beat_mask, chord_attention_mask = chord_mask
			).sample

		elif self.set_from == "pre-trained":
			compressed_latents = self.group_in(noisy_latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
			model_pred = self.unet(
				compressed_latents, timesteps, encoder_hidden_states, 
				encoder_attention_mask=boolean_encoder_mask
			).sample
			model_pred = self.group_out(model_pred.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

		if self.snr_gamma is None:
			loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
		else:
			# Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
			# Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
			snr = self.compute_snr(timesteps)
			mse_loss_weights = (
				torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
			)
			loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
			loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
			loss = loss.mean()

		return loss

	@torch.no_grad()
	def inference(self, prompt, beats, chords,chords_time, inference_scheduler, num_steps=20, guidance_scale=3, num_samples_per_prompt=1, 
				  disable_progress=True):
		# device = self.text_encoder.device
		device = "cpu"
		classifier_free_guidance = guidance_scale > 1.0
		batch_size = len(prompt) * num_samples_per_prompt

		if classifier_free_guidance:
			prompt_embeds, boolean_prompt_mask = self.encode_text_classifier_free(prompt, num_samples_per_prompt)
			encoded_beats, beat_mask = self.encode_beats(beats) #batch, len_beats, dim; batch, len_beats
			encoded_chords, chord_mask = self.encode_chords(chords,chords_time)

			#Nic TODO: check with DEEP
			encoded_beats, beat_mask = torch.cat((encoded_beats, encoded_beats)), torch.cat((beat_mask, beat_mask))
			encoded_chords, chord_mask = torch.cat((encoded_chords, encoded_chords)), torch.cat((chord_mask, chord_mask))
		else:
			prompt_embeds, boolean_prompt_mask = self.encode_text(prompt)
			prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
			boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)
			encoded_beats, beat_mask = self.encode_beats(beats) #batch, len_beats, dim; batch, len_beats
			encoded_chords, chord_mask = self.encode_chords(chords,chords_time)
		print(f"encoded_chords:{encoded_chords.shape}, chord_mask:{chord_mask.shape}, prompt_embeds:{prompt_embeds.shape},boolean_prompt_mask:{boolean_prompt_mask.shape} ")
		inference_scheduler.set_timesteps(num_steps, device=device)
		timesteps = inference_scheduler.timesteps

		num_channels_latents = self.unet.in_channels
		latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, prompt_embeds.dtype, device)

		num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
		progress_bar = tqdm(range(num_steps), disable=disable_progress)

		for i, t in enumerate(timesteps):
			# expand the latents if we are doing classifier free guidance
			latent_model_input = torch.cat([latents] * 2) if classifier_free_guidance else latents
			latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)

			noise_pred = self.unet(
				latent_model_input, t, encoder_hidden_states=prompt_embeds,
				encoder_attention_mask=boolean_prompt_mask, 
				beat_features = encoded_beats, beat_attention_mask = beat_mask, chord_features = encoded_chords,chord_attention_mask = chord_mask
			).sample

			# perform guidance
			if classifier_free_guidance:
				noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
				noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

			# compute the previous noisy sample x_t -> x_t-1
			latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

			# call the callback, if provided
			if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
				progress_bar.update(1)

		if self.set_from == "pre-trained":
			latents = self.group_out(latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
		return latents

	def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents, dtype, device):
		shape = (batch_size, num_channels_latents, 256, 16)
		latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
		# scale the initial noise by the standard deviation required by the scheduler
		latents = latents * inference_scheduler.init_noise_sigma
		return latents

	def encode_text_classifier_free(self, prompt, num_samples_per_prompt):
		# device = self.text_encoder.device
		device = "cpu"
		batch = self.tokenizer(
			prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
		)
		input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

		with torch.no_grad():
			prompt_embeds = self.text_encoder(
				input_ids=input_ids, attention_mask=attention_mask
			)[0]
				
		prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
		attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

		# get unconditional embeddings for classifier free guidance
		uncond_tokens = [""] * len(prompt)

		max_length = prompt_embeds.shape[1]
		uncond_batch = self.tokenizer(
			uncond_tokens, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt",
		)
		uncond_input_ids = uncond_batch.input_ids.to(device)
		uncond_attention_mask = uncond_batch.attention_mask.to(device)

		with torch.no_grad():
			negative_prompt_embeds = self.text_encoder(
				input_ids=uncond_input_ids, attention_mask=uncond_attention_mask
			)[0]
				
		negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
		uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

		# For classifier free guidance, we need to do two forward passes.
		# We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
		prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
		prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
		boolean_prompt_mask = (prompt_mask == 1).to(device)

		return prompt_embeds, boolean_prompt_mask
	
