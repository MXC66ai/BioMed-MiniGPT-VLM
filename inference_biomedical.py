"""
BioMed MiniGPT VLM Interactive Inference Script
Supports single image inference for biomedical image captioning and VQA
"""

import argparse
import os

import torch
from PIL import Image

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="BioMed MiniGPT VLM Inference")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--image", type=str, required=True, help="path to biomedical image.")
    parser.add_argument("--question", type=str, default=None, help="question for VQA mode. If not provided, performs image captioning.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    import random
    import numpy as np
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)

    device = torch.device(f"cuda:{args.gpu_id}")

    # Build model
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    # Build vis_processor
    vis_processor_cfg = cfg.datasets_cfg.biomedical_mixed.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    # Load and process image
    image = Image.open(args.image).convert("RGB")
    image_tensor = vis_processor(image).unsqueeze(0).to(device)

    # Prepare prompt
    if args.question:
        # VQA mode
        prompt = f"<Img><ImageHere></Img> [biomedical_vqa] {args.question} "
        print(f"\n[Biomedical VQA]")
        print(f"Image: {args.image}")
        print(f"Question: {args.question}")
    else:
        # Image captioning mode
        prompt = "<Img><ImageHere></Img> Describe this biomedical image in detail. "
        print(f"\n[Biomedical Image Captioning]")
        print(f"Image: {args.image}")

    # Generate answer
    with torch.no_grad():
        answer = model.generate(
            image_tensor,
            [prompt],
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
            temperature=args.temperature,
            do_sample=False,
        )[0]

    print(f"\nAnswer: {answer.strip()}")
    print("-" * 50)


if __name__ == "__main__":
    main()
