BioMed MiniGPT VLM: Biomedical MiniGPT Vision-Language Model
项目声明：

BioMed MiniGPT VLM（生医通）是一个面向生物医学场景的视觉-语言多模态理解模型。本项目在开源视觉-语言架构（MiniGPT-4）的基础上进行代码修改与用PMC-VQA数据集对模型进行微调，旨在为医学影像理解、生物医学图像描述与视觉问答等任务提供可复现的研究基线。

* 本项目仅用于学术研究目的，不对模型输出的医学内容承担任何责任，模型生成结果不可直接作为临床诊断依据。

* 我们充分尊重原始开源项目的知识产权，完整保留了原始架构的 BSD 3-Clause 许可证。
一、项目概述

BioMed MiniGPT VLM（生医通）支持以下生物医学多模态理解任务：

* 生物医学图像描述（Biomedical Image Captioning）：基于 PMC-OA 数据集，自动生成医学影像的文本描述

* 生物医学视觉问答（Biomedical VQA）：基于 PMC-VQA / Slake 1.0 数据集，回答医学影像相关的专业问题

### 核心特性

| 特性     | 说明                                            |
| ------ | --------------------------------------------- |
| 轻量架构   | 基于 Vicuna 7B + EVA-ViT 视觉编码器，无需 Q-Former，单卡可训 |
| 多数据集支持 | 同时支持 PMC-OA（图像描述）和 PMC-VQA（视觉问答）的联合训练         |
| 灵活配置   | 支持单数据集训练、混合训练、两阶段微调等多种训练策略                    |
| 中文适配   | 支持 PMC-VQA 中的中英文双语医学问答                        |

***

## 二、数据集说明

### 2.1 数据集路径配置

| 数据集                 | 训练集                                    | 测试集                                   | 验证集                                       | 图片根目录                       |
| ------------------- | -------------------------------------- | ------------------------------------- | ----------------------------------------- | --------------------------- |
| **PMC-OA**          | `BioMed-MiniGPT-VLM\PMC-OA\train.jsonl`          | `BioMed-MiniGPT-VLM\PMC-OA\test.jsonl`          | `BioMed-MiniGPT-VLM\PMC-OA\valid.jsonl`             | `BioMed-MiniGPT-VLM\PMC-OA\images`    |
| **PMC-VQA (Slake)** | `BioMed-MiniGPT-VLM\Slake1.0\train.json` | `BioMed-MiniGPT-VLM\Slake1.0\test.json` | `BioMed-MiniGPT-VLM\PMC-VQA\Slake1.0\validate.json` | `BioMed-MiniGPT-VLM\PMC-VQA\Slake1.0` |

> 数据集配置文件位于 `BioMed-MiniGPT-VLM/configs/datasets/` 目录下，可根据实际路径修改。

### 2.2 数据格式说明

**PMC-OA（JSONL 格式）**：

```json
{"image": "PMC212319_Fig3_4.jpg", "caption": "...", "pmcid": "PMC212319", "url_name": "..."}
```

**PMC-VQA（JSON 格式）**：

```json
{"img_name": "xmlab1/source.jpg", "question": "What modality is used?", "answer": "MRI", ...}
```

***

## 三、环境准备

### 3.1 推荐环境

| 项目      | 要求                     |
| ------- | ---------------------- |
| Python  | 3.9                    |
| PyTorch | 2.0.0 + CUDA 11.7/11.8 |
| GPU     | 显存 ≥ 16GB（推荐）          |

### 3.2 安装依赖

```bash
cd BioMed-MiniGPT-VLM
conda create -n biomedMiniGptv python=3.9
conda activate biomedMiniGptv
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

或直接使用项目自带的配置：

```bash
conda env create -f environment.yml
conda activate biomedMiniGptv
```

### 3.3 准备模型权重

**Vicuna 7B 权重**：

1. 下载 Vicuna V0 7B 模型权重
2. 将权重放置在 `BioMed-MiniGPT-VLM\vicuna_7b` 目录
3. 修改模型配置文件中的路径：`BioMed-MiniGPT-VLM/configs/models/minigpt4_vicuna7b_biomedical.yaml`

```yaml
llama_model: "BioMed-MiniGPT-VLM/vicuna_7b"
```

**预训练检查点**（可选）：

* 下载 MiniGPT-4 (Vicuna 7B) 官方预训练检查点

* 在训练配置中设置 `ckpt` 路径以进行微调

***

## 四、代码说明


### 4.1 关键设计决策

* **关闭 Q-Former**（`has_qformer: False`）：在 Vicuna 7B 配置下，直接使用视觉编码器输出的特征，通过投影层对齐到 LLM 隐空间，降低显存占用，适配单卡训练场景。

* **生物医学指令模板**：设计了专用的 prompt template，区分图像描述任务和视觉问答任务。

* **缺失图片自动过滤**：数据集加载器自动检测图片是否存在，仅加载有效样本，避免训练中断。

***

## 五、训练指南

### 5.1 第一阶段：生物医学多模态预训练

```bash
python train_biomedical.py \
    --cfg-path configs/minigpt4_vicuna7b_biomedical_stage1.yaml
```

**配置说明**：

* 数据集：`biomedical_mixed`（PMC-OA + PMC-VQA 联合训练）

* 学习率：1e-4 → 8e-5

* 批次大小：8

* 训练轮数：10 epoch

* 输出目录：`output/minigpt4_vicuna7b_biomedical_stage1`

### 5.2 第二阶段：高质量生物医学微调

```bash
python train_biomedical.py \
    --cfg-path train_configs/minigpt4_vicuna7b_biomedical_stage2.yaml
```

**配置说明**：

* 从 Stage 1 最佳检查点加载

* 更低学习率：3e-5 → 1e-5

* 更小批次：4

* 输出目录：`output/minigpt4_vicuna7b_biomedical_stage2`

### 5.3 单独训练 PMC-OA

修改 `configs/minigpt4_vicuna7b_biomedical_stage1.yaml` 中的 datasets 部分：

```yaml
datasets:
  pmc_oa:
    batch_size: 8
    ...
```

### 5.4 单独训练 PMC-VQA

```yaml
datasets:
  pmc_vqa:
    batch_size: 8
    ...
```

***

## 六、评估指南

### 6.1 评估 PMC-VQA（测试集）

```bash
python eval_biomedical.py \
    --cfg-path BioMed-MiniGPT-VLM/configs/models/minigpt4_vicuna7b_biomedical.yaml \
    --dataset pmc_vqa \
    --split test \
    --batch-size 8 \
    --max-new-tokens 100 \
    --output-dir output/biomedical_eval
```

### 6.2 评估 PMC-OA（测试集）

```bash
python eval_biomedical.py \
    --cfg-path BioMed-MiniGPT-VLM/configs/models/minigpt4_vicuna7b_biomedical.yaml \
    --dataset pmc_oa \
    --split test \
    --batch-size 8 \
    --max-new-tokens 200 \
    --output-dir output/biomedical_eval
```

### 6.3 评估结果输出

评估结果保存在 `output/biomedical_eval/` 目录：

* `pmc_vqa_test_results.json`：包含准确率、每题预测结果与匹配状态

* `pmc_oa_test_results.json`：包含生成的图像描述

***

## 七、推理指南

### 7.1 生物医学图像描述

```bash
python inference_biomedical.py \
    --cfg-path BioMed-MiniGPT-VLM/configs/models/minigpt4_vicuna7b_biomedical.yaml \
    --image path/to/biomedical_image.jpg \
    --max-new-tokens 200
```

### 7.2 生物医学视觉问答

```bash
python inference_biomedical.py \
    --cfg-path BioMed-MiniGPT-VLM/configs/models/minigpt4_vicuna7b_biomedical.yaml \
    --image path/to/medical_image.jpg \
    --question "What modality is used to take this image?" \
    --max-new-tokens 50
```

***

## 八、模型配置说明

### 8.1 Vicuna 7B 生物医学配置

文件：`BioMed-MiniGPT-VLM/configs/models/minigpt4_vicuna7b_biomedical.yaml`

```yaml
model:
  arch: minigpt4
  model_type: pretrain_vicuna0
  image_size: 224
  vit_precision: "fp16"
  freeze_vit: True
  has_qformer: False          # 关闭 Q-Former，减少显存占用
  llama_model: "D:/SDURA/MiniGPT-4/vicuna_7b"
  max_txt_len: 160
  end_sym: "###"
  prompt_path: "prompts/alignment.txt"
  prompt_template: '###Human: {} ###Assistant: '
```

### 8.2 关键参数说明

| 参数             | 说明            | 建议值                         |
| -------------- | ------------- | --------------------------- |
| `has_qformer`  | 是否使用 Q-Former | `False`（7B 推荐关闭以降低显存）       |
| `freeze_vit`   | 是否冻结视觉编码器     | `True`                      |
| `max_txt_len`  | 最大文本长度        | 160                         |
| `batch_size`   | 批次大小          | 4-8（取决于 GPU 显存）             |
| `init_lr`      | 初始学习率         | 1e-4（Stage 1），3e-5（Stage 2） |
| `low_resource` | 启用 8-bit 量化   | 显存不足时设为 `True`              |

***

## 九、致谢与引用

**BioMed MiniGPT VLM（生医通）** 基于以下优秀开源项目构建，我们向原始作者表示衷心感谢：

* [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)：视觉-语言对齐架构

* [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)：视觉编码器与 Q-Former 设计

* [Vicuna](https://github.com/lm-sys/FastChat)：开源大语言模型

* [PMC-OA](https://huggingface.co/datasets/axiong/pmc_oa) 与 [PMC-VQA](https://github.com/xiaoman-zhang/PMC-VQA) / [Slake](https://github.com/SLAKE-SLAKE/Slake)：生物医学数据集

如需引用原始 MiniGPT-4 论文：

```bibtex
@article{zhu2023minigpt,
  title={MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models},
  author={Zhu, Deyao and Chen, Jun and Shen, Xiaoqian and Li, Xiang and Elhoseiny, Mohamed},
  journal={arXiv preprint arXiv:2304.10592},
  year={2023}
}
```

如需引用 Slake（PMC-VQA）数据集：

```bibtex
@inproceedings{liu2021slake,
  title={Slake: A dataset of visual question answering over biomedical data},
  author={Liu, Bo and Zhan, Li-Ming and Xu, Lei and Ma, Lianli and Yang, Yi and Wu, Xing},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2021}
}
```

***

## 十一、许可证

本项目代码遵循与原始 MiniGPT-4 相同的 **BSD 3-Clause License**。项目中大量代码基于 [LAVIS](https://github.com/salesforce/LAVIS) 构建，亦遵循 BSD 3-Clause License。

**免责声明**：本模型仅供学术研究使用，不应用于临床诊断或医疗决策。模型输出的医学内容可能存在错误，使用者需自行承担风险。
