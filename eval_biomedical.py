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

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from minigpt4.datasets.datasets.biomedical_datasets import PMCOADataset, PMCVQADataset


def parse_args():
    parser = argparse.ArgumentParser(description="BioMed MiniGPT VLM Evaluation")
    parser.add_argument("--cfg-path", required=True, help="path to model configuration file.")
    parser.add_argument("--dataset-cfg", default=None, help="path to dataset configuration file. Auto-detected if not provided.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--dataset", type=str, default="pmc_vqa", choices=["pmc_oa", "pmc_vqa"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="output/biomedical_eval")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="if > 0, evaluate on a deterministic random subset of this size")
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


def load_model_and_config(args):
    """Load model from config and optionally override checkpoint."""
    cfg = Config(args)
    setup_seeds(cfg)

    device = torch.device(f"cuda:{args.gpu_id}")

    # Build model
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    print(f"[BioMed MiniGPT VLM] cuda_available={torch.cuda.is_available()} "
          f"model_device={next(model.parameters()).device} "
          f"gpu_mem_allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
          f"gpu_mem_reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")

    # Build vis_processor from config
    vis_processor_cfg = cfg.config.preprocess.vis_processor.eval
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    # Build text_processor from config
    text_processor_cfg = cfg.config.preprocess.text_processor.eval
    text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)

    return model, vis_processor, text_processor, device, cfg


def build_evaluation_dataset(dataset_cfg_path, dataset_name, split, vis_processor, text_processor):
    """Build dataset for evaluation using the dataset config."""
    import yaml
    from omegaconf import OmegaConf

    # Load dataset config
    with open(dataset_cfg_path, 'r') as f:
        dataset_yaml = yaml.safe_load(f)

    # Get dataset configuration
    if dataset_name == "pmc_vqa":
        dataset_info = dataset_yaml['datasets']['pmc_vqa']
        annotations_path = dataset_info['build_info']['annotations'][split]['storage']
        images_path = dataset_info['build_info']['images']['storage']
        dataset = PMCVQADataset(vis_processor, text_processor, images_path, annotations_path)

    elif dataset_name == "pmc_oa":
        dataset_info = dataset_yaml['datasets']['pmc_oa']
        annotations_path = dataset_info['build_info']['annotations'][split]['storage']
        images_path = dataset_info['build_info']['images']['storage']
        dataset = PMCOADataset(vis_processor, text_processor, images_path, annotations_path)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"[BioMed MiniGPT VLM] Loaded {dataset_name} {split} split: {len(dataset)} samples")
    return dataset


def evaluate_pmc_oa(model, dataset, args, device, prompt_template):
    """Evaluate PMC-OA: image captioning task"""
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    results = []
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating PMC-OA"):
            images = batch["image"].to(device)
            image_ids = batch["image_id"]
            instructions = batch["instruction_input"]

            texts = [prompt_template.format(inst) for inst in instructions]
            try:
                # Batched generation: one forward pass for the whole batch
                answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens)
            except Exception as e:
                print(f"[WARN] batched generate failed ({e}), falling back to per-sample")
                answers = []
                for i in range(len(images)):
                    try:
                        answer = model.generate(images[i:i+1], [texts[i]], max_new_tokens=args.max_new_tokens)[0]
                    except Exception as e2:
                        print(f"[WARN] generate failed for image_id={image_ids[i]}: {e2}")
                        answer = "Error generating caption"
                    answers.append(answer)

            for answer, image_id in zip(answers, image_ids):
                results.append({
                    "image_id": image_id,
                    "generated_caption": str(answer).split("###")[0].strip(),
                })
            total_samples += len(images)

    print(f"\n[BioMed MiniGPT VLM] PMC-OA Evaluation completed: {total_samples} samples evaluated.")
    return results


def evaluate_pmc_vqa(model, dataset, args, device, prompt_template):
    """Evaluate PMC-VQA: visual question answering task"""
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    results = []
    correct = 0
    total = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating PMC-VQA"):
            images = batch["image"].to(device)
            questions = batch["question"]
            instructions = batch["instruction_input"]
            gt_answers = batch["answer"]
            image_ids = batch["image_id"]

            texts = [prompt_template.format(inst) for inst in instructions]
            try:
                # Batched generation: one forward pass for the whole batch
                answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens)
            except Exception as e:
                print(f"[WARN] batched generate failed ({e}), falling back to per-sample")
                answers = []
                for i in range(len(images)):
                    try:
                        answer = model.generate(images[i:i+1], [texts[i]], max_new_tokens=args.max_new_tokens)[0]
                    except Exception as e2:
                        print(f"[WARN] generate failed for image_id={image_ids[i]}: {e2}")
                        answer = "Error generating answer"
                    answers.append(answer)

            for answer, gt, q, img_id in zip(answers, gt_answers, questions, image_ids):
                pred = str(answer).split("###")[0].strip().lower()
                gt_lower = str(gt).strip().lower()

                # Exact match for closed-ended questions, contains match for open-ended
                is_match = pred == gt_lower or gt_lower in pred or pred in gt_lower
                if is_match:
                    correct += 1
                total += 1

                results.append({
                    "image_id": str(img_id),
                    "question": q,
                    "ground_truth": str(gt),
                    "predicted": str(answer).split("###")[0].strip(),
                    "match": is_match,
                })
            total_samples += len(images)

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n[BioMed MiniGPT VLM] PMC-VQA Evaluation completed: {total_samples} samples evaluated.")
    print(f"[BioMed MiniGPT VLM] Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return results, accuracy


def main():
    args = parse_args()

    # Auto-detect dataset config if not provided
    if args.dataset_cfg is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.dataset_cfg = os.path.join(
            script_dir,
            "minigpt4",
            "configs",
            "datasets",
            "biomedical",
            f"{args.dataset}.yaml"
        )
        print(f"[BioMed MiniGPT VLM] Auto-detected dataset config: {args.dataset_cfg}")

    print("=" * 60)
    print("BioMed MiniGPT VLM (生医通) Evaluation")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Dataset config: {args.dataset_cfg}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("=" * 60)

    # Load model
    print("\n[BioMed MiniGPT VLM] Loading model...")
    model, vis_processor, text_processor, device, cfg = load_model_and_config(args)
    print("[BioMed MiniGPT VLM] Model loaded successfully!")

    # Prompt template from config (matches Stage-2 training format)
    prompt_template = cfg.config.model.get("prompt_template", "###Human: {} ###Assistant: ")
    print(f"[BioMed MiniGPT VLM] Using prompt template: {prompt_template!r}")

    # Build dataset
    print(f"\n[BioMed MiniGPT VLM] Loading {args.dataset} dataset...")
    eval_dataset = build_evaluation_dataset(args.dataset_cfg, args.dataset, args.split, vis_processor, text_processor)

    # Optional deterministic subsample for very large test sets (e.g. PMC-OA)
    if args.max_samples > 0 and len(eval_dataset) > args.max_samples:
        from torch.utils.data import Subset
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(eval_dataset))[:args.max_samples].tolist()
        eval_dataset = Subset(eval_dataset, indices)
        print(f"[BioMed MiniGPT VLM] Subsampled {args.max_samples} samples (seed=42) for evaluation.")

    # Run evaluation
    os.makedirs(args.output_dir, exist_ok=True)

    suffix = f"_sub{args.max_samples}" if args.max_samples > 0 else ""

    if args.dataset == "pmc_oa":
        results = evaluate_pmc_oa(model, eval_dataset, args, device, prompt_template)
        output_file = os.path.join(args.output_dir, f"pmc_oa_{args.split}{suffix}_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[BioMed MiniGPT VLM] Results saved to {output_file}")

    elif args.dataset == "pmc_vqa":
        results, accuracy = evaluate_pmc_vqa(model, eval_dataset, args, device, prompt_template)
        output_file = os.path.join(args.output_dir, f"pmc_vqa_{args.split}_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "accuracy": accuracy,
                "total_samples": len(results),
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"[BioMed MiniGPT VLM] Results saved to {output_file}")

    print("\n" + "=" * 60)
    print("BioMed MiniGPT VLM (生医通) evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
