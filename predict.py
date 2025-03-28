import io
import os
import time
import subprocess

import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from scepter.modules.transform.io import pillow_convert
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS

from inference.ace_plus_diffusers import ACEPlusDiffuserInference


def download_weights(url, dest, file=False):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if not file:
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


fs_list = [
    Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "HttpFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "LocalFs", "TEMP_DIR": "./cache"}, load=False),
]

for one_fs in fs_list:
    FS.init_fs_client(one_fs)


def run_one_case(pipe,
                input_image = None,
                input_mask = None,
                input_reference_image = None,
                save_path = "examples/output/example.png",
                instruction = "",
                output_h = 1024,
                output_w = 1024,
                seed = -1,
                sample_steps = None,
                guide_scale = None,
                repainting_scale = None,
                model_path = None,
                **kwargs):
    if input_image is not None:
        input_image = Image.open(io.BytesIO(FS.get_object(input_image)))
        input_image = pillow_convert(input_image, "RGB")
    if input_mask is not None:
        input_mask = Image.open(io.BytesIO(FS.get_object(input_mask)))
        input_mask = pillow_convert(input_mask, "L")
    if input_reference_image is not None:
        input_reference_image = Image.open(io.BytesIO(FS.get_object(input_reference_image)))
        input_reference_image = pillow_convert(input_reference_image, "RGB")

    image, seed = pipe(
        reference_image=input_reference_image,
        edit_image=input_image,
        edit_mask=input_mask,
        prompt=instruction,
        output_height=output_h,
        output_width=output_w,
        sampler='flow_euler',
        sample_steps=sample_steps or pipe.input.get("sample_steps", 28),
        guide_scale=guide_scale or pipe.input.get("guide_scale", 50),
        seed=seed,
        repainting_scale=repainting_scale or pipe.input.get("repainting_scale", 1.0),
        lora_path = model_path
    )
    with FS.put_to(save_path) as local_path:
        image.save(local_path)
    return local_path, seed

MODEL_CACHE = "FLUX.1-dev"
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"

SUBJECT_ACE_LORA_CACHE = "subject.safetensors"
SUBJECT_ACE_LORA_URL = "https://huggingface.co/ali-vilab/ACE_Plus/resolve/main/subject/comfyui_subject_lora16.safetensors?download=true"


class CogPredictor(BasePredictor):
    def setup(self) -> None:
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, ".")
            print("Downloaded flux weights")

        if not os.path.exists(SUBJECT_ACE_LORA_CACHE):
            download_weights(SUBJECT_ACE_LORA_URL, SUBJECT_ACE_LORA_CACHE)
            print("Downloaded subject lora weights")

        diffuser_pipeline_config = "./config/ace_plus_diffusers_infer.yaml"
        self.pipe_cfg = Config(load=True, cfg_file=diffuser_pipeline_config)
        print("config pipeline loaded")

        self.pipe = ACEPlusDiffuserInference()
        print("ACEPlusDiffuserInference loaded")
        self.pipe.init_from_cfg(self.pipe_cfg)
        print("ACEPlusDiffuserInference initialized")

    @torch.inference_mode()
    def predict(
            self,
            prompt: str = Input(
                description="Prompt",
                default=""
            ),
            input_reference_image: Path = Input(
                description="Input reference image",
                default=None
            ),
            input_image: Path = Input(
                description="Input image",
                default=None
            ),
            input_mask: Path = Input(
                description="Input mask",
                default=None
            ),
            task_type: str = Input(
                description="Task type. Currently only subject mode is supported",
                choices=['portrait', 'subject'],
                default="subject"
            ),
            guidance_scale: float = Input(
                description="Guidance scale",
                default=50,
                ge=0,
                le=600
            ),
            repaint_scale: float = Input(
                description="The repainting scale for content filling generation!",
                default=1,
                ge=0,
                le=10
            ),
            seed: int = Input(
                description="Random seed. Set for reproducible generation",
                default=42
            ),
            width: int = Input(
                description="Width of the output image",
                default=1024
            ),
            height: int = Input(
                description="Height of the output image",
                default=1024
            ),
            num_inference_steps: int = Input(
                description="Number of inference steps",
                ge=1, le=50, default=28,
            ),
    ) -> Path:
        model_path = SUBJECT_ACE_LORA_URL

        # TODO: make it dynamic
        save_path = "output.jpg"
        params = {
            "input_image": input_image,
            "input_mask": input_mask,
            "input_reference_image": input_reference_image,
            "save_path": save_path,
            "instruction": prompt,
            "output_h": height,
            "output_w": width,
            "sample_steps": num_inference_steps,
            "guide_scale": guidance_scale,
            "repainting_scale": repaint_scale,
            "model_path": model_path,
        }
        local_path, seed = run_one_case(self.pipe, **params)
        return Path(local_path)
