"""
BioMed MiniGPT VLM Evaluation Script
Supports PMC-OA (image captioning) and PMC-VQA (visual question answering) evaluation
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

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

from minigpt4.datasets.datasets.biomedical_datasets import PMCOADataset, PMCVQADataset


def parse_args():
    parser = argparse.ArgumentParser(description="BioMed MiniGPT VLM Evaluation")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--dataset", type=str, default="pmc_vqa", choices=["pmc_oa", "pmc_vqa", "biomedical_mixed"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="output/biomedical_eval")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_pmc_oa(model, vis_processor, dataset, args):
    """Evaluate PMC-OA: image captioning task"""
    device = torch.device(f"cuda:{args.gpu_id}")
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = []
    total_samples = 0

    # Biomedical captioning prompt
    prompt_template = "###Human: Describe this biomedical image in detail. ###Assistant: "

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating PMC-OA"):
            images = batch["image"].to(device)
            image_ids = batch["image_id"]

            # Prepare texts with image placeholders
            texts = [f"<Img><ImageHere></Img> {prompt_template}"] * len(images)

            answers = model.generate(
                images,
                texts,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                temperature=1.0,
                do_sample=False,
            )

            for answer, image_id in zip(answers, image_ids):
                results.append({
                    "image_id": image_id,
                    "generated_caption": answer.strip(),
                })
            total_samples += len(images)

    print(f"\nPMC-OA Evaluation completed: {total_samples} samples evaluated.")
    return results


def evaluate_pmc_vqa(model, vis_processor, dataset, args):
    """Evaluate PMC-VQA: visual question answering task"""
    device = torch.device(f"cuda:{args.gpu_id}")
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = []
    correct = 0
    total = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating PMC-VQA"):
            images = batch["image"].to(device)
            questions = batch["question"]
            gt_answers = batch["answer"]
            image_ids = batch["image_id"]

            # Prepare texts with biomedical VQA instruction
            texts = [f"<Img><ImageHere></Img> [biomedical_vqa] {q} " for q in questions]

            answers = model.generate(
                images,
                texts,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                temperature=1.0,
                do_sample=False,
            )

            for answer, gt, q, img_id in zip(answers, gt_answers, questions, image_ids):
                pred = answer.strip().lower()
                gt_lower = str(gt).strip().lower()

                # Exact match for closed-ended questions, contains match for open-ended
                if pred == gt_lower or gt_lower in pred or pred in gt_lower:
                    correct += 1
                total += 1

                results.append({
                    "image_id": img_id,
                    "question": q,
                    "ground_truth": gt,
                    "predicted": answer.strip(),
                    "match": pred == gt_lower or gt_lower in pred or pred in gt_lower,
                })
            total_samples += len(images)

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nPMC-VQA Evaluation completed: {total_samples} samples evaluated.")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return results, accuracy


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

    # Build vis_processor
    vis_processor_cfg = cfg.datasets_cfg.biomedical_mixed.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    # Determine dataset configuration based on args.dataset
    dataset_name = args.dataset
    split = args.split

    # Build dataset
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)

    if dataset_name not in datasets:
        print(f"Dataset {dataset_name} not found in config. Available: {list(datasets.keys())}")
        return

    dataset = datasets[dataset_name]
    if split not in dataset:
        print(f"Split {split} not found in dataset {dataset_name}. Available: {list(dataset.keys())}")
        return

    eval_dataset = dataset[split]
    print(f"Evaluating {dataset_name} - {split} split: {len(eval_dataset)} samples")

    # Run evaluation
    os.makedirs(args.output_dir, exist_ok=True)

    if dataset_name == "pmc_oa":
        results = evaluate_pmc_oa(model, vis_processor, eval_dataset, args)
        output_file = os.path.join(args.output_dir, f"pmc_oa_{split}_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")

    elif dataset_name == "pmc_vqa":
        results, accuracy = evaluate_pmc_vqa(model, vis_processor, eval_dataset, args)
        output_file = os.path.join(args.output_dir, f"pmc_vqa_{split}_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "accuracy": accuracy,
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")

    elif dataset_name == "biomedical_mixed":
        # Evaluate VQA portion
        vqa_results, vqa_accuracy = evaluate_pmc_vqa(model, vis_processor, eval_dataset, args)
        output_file = os.path.join(args.output_dir, f"biomedical_mixed_{split}_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "vqa_accuracy": vqa_accuracy,
                "results": vqa_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")

    print("\nBioMed MiniGPT VLM evaluation completed!")


if __name__ == "__main__":
    main()
