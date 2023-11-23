<div align="center">

# Mustango: Toward Controllable Text-to-Music Generation

[Demo](https://huggingface.co/spaces/declare-lab/mustango) | [Model](https://huggingface.co/declare-lab/mustango) | [Website and Examples](https://amaai-lab.github.io/mustango/) | [Paper](https://arxiv.org/abs/2311.08355) | [Dataset](https://huggingface.co/datasets/amaai-lab/MusicBench)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/declare-lab/mustango)
</div>

Meet Mustango, an exciting addition to the vibrant landscape of Multimodal Large Language Models designed for controlled music generation. Mustango leverages Latent Diffusion Model (LDM), Flan-T5, and musical features to do the magic!

🔥 Live demo available on [Replicate](https://replicate.com/declare-lab/mustango) and [HuggingFace](https://huggingface.co/spaces/declare-lab/mustango).

<div align="center">
  <img src="img/mustango.jpg" width="500"/>
</div>


## Quickstart Guide

Generate music from a text prompt:

```python
import IPython
import soundfile as sf
from mustango import Mustango

model = Mustango("declare-lab/mustango")

prompt = "This is a new age piece. There is a flute playing the main melody with a lot of staccato notes. The rhythmic background consists of a medium tempo electronic drum beat with percussive elements all over the spectrum. There is a playful atmosphere to the piece. This piece can be used in the soundtrack of a children's TV show or an advertisement jingle."

music = model.generate(prompt)
sf.write(f"{prompt}.wav", audio, samplerate=16000)
IPython.display.Audio(data=music, rate=16000)
```

## Installation

```bash
git clone https://github.com/AMAAI-Lab/mustango
cd mustango
pip install -r requirements.txt
cd diffusers
pip install -e .
```

## Datasets

The [MusicBench](https://huggingface.co/datasets/amaai-lab/MusicBench) dataset contains 52k music fragments with a rich music-specific text caption. 
## Subjective Evaluation by Expert Listeners

| **Model** | **Dataset** | **Pre-trained** | **Overall Match** ↑ | **Chord Match** ↑ | **Tempo Match** ↑ | **Audio Quality** ↑ | **Musicality** ↑ | **Rhythmic Presence and Stability** ↑ | **Harmony and Consonance** ↑ |
|-----------|-------------|:-----------------:|:-----------:|:-----------:|:-----------:|:----------:|:----------:|:----------:|:----------:|
| Tango     | MusicCaps   | ✓               | 4.35      | 2.75      | 3.88      | 3.35     | 2.83     | 3.95     | 3.84     |
| Tango     | MusicBench  | ✓               | 4.91      | 3.61      | 3.86      | 3.88     | 3.54     | 4.01     | 4.34     |
| Mustango  | MusicBench  | ✓               | 5.49      | 5.76      | 4.98      | 4.30     | 4.28     | 4.65     | 5.18     |
| Mustango  | MusicBench  | ✗               | 5.75      | 6.06      | 5.11      | 4.80     | 4.80     | 4.75     | 5.59     |




## Training

We use the `accelerate` package from Hugging Face for multi-gpu training. Run `accelerate config` from terminal and set up your run configuration by the answering the questions asked.

You can now train **Mustango** on the MusicBench dataset using:

```bash
accelerate launch train.py \
--text_encoder_name="google/flan-t5-large" \
--scheduler_name="stabilityai/stable-diffusion-2-1" \
--unet_model_config="configs/diffusion_model_config_munet.json" \
--model_type Mustango --freeze_text_encoder --uncondition_all --uncondition_single \
--drop_sentences --random_pick_text_column --snr_gamma 5 \
```

The `--model_type` flag allows to choose either Mustango, or Tango to be trained with the same code. However, do note that you also need to change `--unet_model_config` to the relevant config: diffusion_model_config_munet for Mustango; diffusion_model_config for Tango.

The arguments `--uncondition_all`, `--uncondition_single`, `--drop_sentences` control the dropout functions as per Section 5.2 in our paper. The argument of `--random_pick_text_column` allows to randomly pick between two input text prompts - in the case of MusicBench, we pick between ChatGPT rephrased captions and original enhanced MusicCaps prompts, as depicted in Figure 1 in our paper.

Recommended training time from scratch on MusicBench is at least 40 epochs.


## Model Zoo

We have released the following models:

Mustango Pretrained: https://huggingface.co/declare-lab/mustango-pretrained


Mustango: https://huggingface.co/declare-lab/mustango


## Citation
Please consider citing the following article if you found our work useful:
```
@misc{melechovsky2023mustango,
      title={Mustango: Toward Controllable Text-to-Music Generation}, 
      author={Jan Melechovsky and Zixun Guo and Deepanway Ghosal and Navonil Majumder and Dorien Herremans and Soujanya Poria},
      year={2023},
      eprint={2311.08355},
      archivePrefix={arXiv},
}
```
