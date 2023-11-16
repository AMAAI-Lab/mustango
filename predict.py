# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import subprocess

subprocess.run("cd diffusers && pip install . && cd ..", shell=True, check=True)

import soundfile as sf
from cog import BasePredictor, Input, Path
from mustango import Mustango


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        cache_dir = "model_cache"
        local_files_only = True  # set to True if models are cached in cache_dir
        self.model = Mustango(
            "declare-lab/mustango", cache_dir=cache_dir, local_files_only=local_files_only
        )

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt.",
            default="This is a new age piece. There is a flute playing the main melody with a lot of staccato notes. The rhythmic background consists of a medium tempo electronic drum beat with percussive elements all over the spectrum. There is a playful atmosphere to the piece. This piece can be used in the soundtrack of a children's TV show or an advertisement jingle.",
        ),
        steps: int = Input(description="inferene steps", default=100),
        guidance: float = Input(description="guidance scale", default=3),
    ) -> Path:
        """Run a single prediction on the model"""

        music = self.model.generate(prompt, steps=steps, guidance=guidance)
        out = "/tmp/output.wav"
        sf.write(out, music, samplerate=16000)
        return Path(out)
