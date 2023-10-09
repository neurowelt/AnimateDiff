import os
import argparse
import datetime
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from omegaconf import OmegaConf

from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .models.unet import UNet3DConditionModel
from .pipelines.pipeline_animation import AnimationPipeline
from .utils.util import save_videos_grid, load_weights, tensor2vid
from diffusers.utils.import_utils import is_xformers_available


device = "cuda" if torch.cuda.is_available() else "cpu"

def create_args(**kwargs):

    """
    Create args object for generate.
    """

    if kwargs.get("config", None) is None:
        raise ValueError("config is required.")
    if kwargs.get("pretrained_model_path", None) is None:
        raise ValueError("pretrained_model_path is required.")
    if kwargs.get("inference_config", None) is None:
        raise ValueError("inference_config is required.")
    if kwargs.get("W", None) is None:
        raise ValueError("W is required.")
    if kwargs.get("H", None) is None:
        raise ValueError("H is required.")

    args = SimpleNamespace(
        **kwargs
    )

    return args


def generate(
    args: SimpleNamespace,
    savedir: Optional[str] = None,
):
    
    """
    Generate AnimateDiff animations.
    """

    if not savedir:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = f"samples/{Path().stem}-{time_str}"
    
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    config = OmegaConf.load(args.config)
    samples = []
    frames = []
    paths = []
    sample_idx = 0
    for _, (config_key, model_config) in enumerate(list(config.items())):
        
        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:
            inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))

            ### >>> create validation pipeline >>> ###
            tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
            vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")            
            unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

            if is_xformers_available(): 
                unet.enable_xformers_memory_efficient_attention()

            pipeline = AnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            ).to(device)

            pipeline, controlnet, down_features, mid_features = load_weights(
                pipeline,
                args,
                device,
                # motion module
                motion_module_path         = motion_module,
                motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
                # image layers
                dreambooth_model_path      = model_config.get("dreambooth_path", ""),
                lora_model_path            = model_config.get("lora_model_path", ""),
                lora_alpha                 = model_config.get("lora_alpha", 0.8),
            ).to(device)

            # Prepare prompts
            _prompts = getattr(args, "prompts", None)
            if _prompts is None:
                prompts = model_config.prompt
            else:
                if not isinstance(_prompts, list):
                    raise TypeError("prompts must be a list.")
                prompts = _prompts

            n_prompts = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
            
            random_seeds = model_config.get("seed", [-1])
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
            
            config[config_key].random_seed = []

            init_image = getattr(args, "I", None)

            for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
                
                # manually set random seed for reproduction
                if random_seed != -1: torch.manual_seed(random_seed)
                else: torch.seed()
                config[config_key].random_seed.append(torch.initial_seed())

                if controlnet is not None:
                    down_features, mid_features = controlnet(model_config.control.video_path, prompt, n_prompt, random_seed)
                
                print(f"current seed: {torch.initial_seed()}")
                print(f"sampling {prompt} ...")
                sample = pipeline(
                    prompt,
                    negative_prompt     = n_prompt,
                    num_inference_steps = model_config.steps,
                    guidance_scale      = model_config.guidance_scale,
                    width               = args.W,
                    height              = args.H,
                    video_length        = args.L,
                    init_image          = init_image,
                    down_block_control  = down_features, 
                    mid_block_control   = mid_features,
                ).videos
                samples.append(sample)
                clip = tensor2vid(sample)
                frames.append(clip)

                # Continue from last frame
                if getattr(args, "C", False):
                    init_image = clip[-1]

                prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                path = f"{savedir}/sample/{sample_idx}-{prompt}.gif"
                paths.append(path)
                save_videos_grid(sample, path)
                print(f"save to {savedir}/sample/{prompt}.gif")
                
                sample_idx += 1

    return samples, frames, paths