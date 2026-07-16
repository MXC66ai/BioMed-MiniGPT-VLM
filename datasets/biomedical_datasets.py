"""
BioMed MiniGPT VLM Datasets modified from MiniGPT-4 and MiniGPT-v2.
Supports PMC-OA (image-caption pairs) and PMC-VQA (visual question answering)
"""

import json
import os
from PIL import Image
from torch.utils.data import Dataset

from minigpt4.datasets.datasets.base_dataset import BaseDataset


class PMCOADataset(Dataset):
    """
    PMC-OA dataset for biomedical image-caption alignment.
    Data format (jsonl): {"image": "PMCxxx_FigY_Z.jpg", "caption": "...", "pmcid": "PMCxxx", "url_name": "..."}
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.annotation = []
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.annotation.append(json.loads(line))

        # Filter out samples with missing images
        self.valid_indices = []
        for idx, ann in enumerate(self.annotation):
            img_path = self._get_image_path(ann)
            if os.path.exists(img_path):
                self.valid_indices.append(idx)

        if len(self.valid_indices) == 0:
            print(f"[WARNING] [BioMed MiniGPT VLM] No valid images found in {vis_root}. Please ensure images are downloaded.")
        else:
            print(f"[INFO] [BioMed MiniGPT VLM] PMC-OA: {len(self.valid_indices)}/{len(self.annotation)} samples have valid images.")

    def _get_image_path(self, ann):
        """Resolve image path from annotation."""
        img_name = ann.get("image", "")
        # Try direct path first
        img_path = os.path.join(self.vis_root, img_name)
        if os.path.exists(img_path):
            return img_path
        # Try pmcid subfolder
        pmcid = ann.get("pmcid", "")
        if pmcid:
            img_path = os.path.join(self.vis_root, pmcid, img_name)
            if os.path.exists(img_path):
                return img_path
        # Try url_name
        url_name = ann.get("url_name", "")
        if url_name:
            img_path = os.path.join(self.vis_root, url_name)
            if os.path.exists(img_path):
                return img_path
        return os.path.join(self.vis_root, img_name)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        real_idx = self.valid_indices[index]
        ann = self.annotation[real_idx]

        img_path = self._get_image_path(ann)
        image = Image.open(img_path).convert("RGB")
        image = self.vis_processor(image)

        caption = ann.get("caption", "")
        caption = self.text_processor(caption) if self.text_processor else caption

        # Biomedical instruction prompt
        instruction = "<Img><ImageHere></Img> Describe this biomedical image in detail. "

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": caption,
            "image_id": ann.get("pmcid", str(real_idx)),
        }


class PMCVQADataset(Dataset):
    """
    PMC-VQA (Slake) dataset for biomedical visual question answering.
    Data format (json): [{"img_name": "xmlab1/source.jpg", "question": "...", "answer": "...", ...}]
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        with open(ann_path, 'r', encoding='utf-8') as f:
            self.annotation = json.load(f)

        # Filter out samples with missing images
        self.valid_indices = []
        for idx, ann in enumerate(self.annotation):
            img_path = self._get_image_path(ann)
            if os.path.exists(img_path):
                self.valid_indices.append(idx)

        if len(self.valid_indices) == 0:
            print(f"[WARNING] [BioMed MiniGPT VLM] No valid images found in {vis_root}. Please ensure images are downloaded.")
        else:
            print(f"[INFO] [BioMed MiniGPT VLM] PMC-VQA: {len(self.valid_indices)}/{len(self.annotation)} samples have valid images.")

    def _get_image_path(self, ann):
        """Resolve image path from annotation."""
        img_name = ann.get("img_name", "")
        return os.path.join(self.vis_root, img_name)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        real_idx = self.valid_indices[index]
        ann = self.annotation[real_idx]

        img_path = self._get_image_path(ann)
        image = Image.open(img_path).convert("RGB")
        image = self.vis_processor(image)

        question = ann.get("question", "")
        answer = ann.get("answer", "")

        # Biomedical VQA instruction prompt
        instruction = f"<Img><ImageHere></Img> [biomedical_vqa] {question} "

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": str(answer),
            "image_id": ann.get("qid", str(real_idx)),
            "question": question,
        }


class BiomedicalMixedDataset(Dataset):
    """
    Mixed dataset combining PMC-OA (image-caption) and PMC-VQA (VQA).
    This allows joint training on both biomedical understanding tasks.
    """
    def __init__(self, vis_processor, text_processor, vis_roots, ann_paths, dataset_types):
        """
        vis_roots: list of image root directories
        ann_paths: list of annotation file paths
        dataset_types: list of dataset type strings ('pmc_oa' or 'pmc_vqa')
        """
        self.datasets = []
        self.dataset_offsets = [0]

        for vis_root, ann_path, ds_type in zip(vis_roots, ann_paths, dataset_types):
            if ds_type == 'pmc_oa':
                ds = PMCOADataset(vis_processor, text_processor, vis_root, ann_path)
            elif ds_type == 'pmc_vqa':
                ds = PMCVQADataset(vis_processor, text_processor, vis_root, ann_path)
            else:
                raise ValueError(f"Unknown dataset type: {ds_type}")
            self.datasets.append(ds)
            self.dataset_offsets.append(self.dataset_offsets[-1] + len(ds))

    def __len__(self):
        return self.dataset_offsets[-1]

    def __getitem__(self, index):
        # Find which dataset this index belongs to
        for i in range(len(self.datasets)):
            if index < self.dataset_offsets[i + 1]:
                local_idx = index - self.dataset_offsets[i]
                return self.datasets[i][local_idx]
        raise IndexError(f"Index {index} out of range")
