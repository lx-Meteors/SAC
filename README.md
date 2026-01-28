# Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors

<p align="center">
    <a href="https://arxiv.org/abs/2510.08907"><img src="https://img.shields.io/badge/arXiv-2408.03094-b31b1b.svg" alt="Paper"></a>
    <a href=""><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow" alt="Models"></a>
    <a href=""><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-orange" alt="Datasets"></a>
    <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg" alt="License"></a>
    <a href="https://iclr.cc/"><img src="https://img.shields.io/badge/ICLR-2026%20Poster-blue" alt="Conference"></a>
</p>

<p align="center">
  <b><a href="#-environment">🛠️ Environment</a></b> •
  <b><a href="#-training--evaluation">📖 Training</a></b> •
  <b><a href="#-results">📊 Results</a></b> •
  <b><a href="#-citation">📌 Citation</a></b>

[//]: # (  <b><a href="#-news">🚀 News</a></b> •)

[//]: # (  <b><a href="#-introduction">✨ Introduction</a></b> •)
</p>

---

[//]: # (## 🚀 News)

[//]: # (- **[2025.05.15]** 🎉 Our paper was accepted by **ACL 2025 Main**.)

[//]: # (- **[2024.08.06]** The paper was uploaded to [Arxiv]&#40;https://arxiv.org/abs/2408.03094&#41;.)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## ✨ Introduction)

[//]: # (**500xCompressor** 是一种创新的提示词压缩方法，能够将高达 **500个** 自然语言 Token 压缩为仅 **1个** 特殊 Token。该压缩 Token 可直接用于 **文本还原 &#40;Regeneration&#41;** 或 **下游问答 &#40;QA&#41;** 任务。)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (### 🌟 Key Advantages)

[//]: # (- **极高效率**: 仅需为 LLM 添加 **0.3%** 的额外参数（基于 LoRA）。)

[//]: # (- **即插即用**: 压缩后的 Token 可被原始 LLM 直接识别，无需对 LLM 进行额外微调。)

[//]: # (- **超高压缩比**: 支持从 **6x** 到 **480x** 的压缩范围。)

[//]: # (- **强泛化能力**: 在 **Strictly Unseen** 的文本和数据集上表现优异。)

[//]: # (- **能力保留**: 相比未压缩的 Prompt，保留了 **62.26-72.89%** 的模型能力。)

[//]: # ()
[//]: # (---)

## 🛠️ Environment

```bash
# 创建并激活环境
conda create -n SAC python=3.10.4
conda activate SAC
# 安装依赖
pip install -r requirements.txt
```

## 📦 Models & Datasets

| 类别 (Type) | 资源名称 (Resource Name) | 获取链接 (Access Link) | 检查点 (SAC CheckPoint) |
| :--- | :--- | :--- | :--- |
| **Model** | **Llama-3.2-1B** | [🤗 HF Link](https://huggingface.co/meta-llama/Llama-3.2-1B) |
| **Model** | **Llama-3.2-3B** | [🤗 HF Link](https://huggingface.co/meta-llama/Llama-3.2-3B) |
| **Model** | **Llama-3.1-8B** | [🤗 HF Link](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| **Corpus** | **SlimPajama-6B** (Pre-train) | [🤗 Dataset](https://huggingface.co/datasets/DKYoon/SlimPajama-6B) |
| **Dataset** | **MRQA** (Fine-tune) | [🤗 Dataset](https://huggingface.co/datasets/mrqa-workshop/mrqa) |

## 📖 Training & Evaluation

### Config 配置

修改 `experiment/sac_experiment/config.json` 文件：
```bash
"model_id": "your_model_path",
"dataset_repo": "your_data_path/DKYoon/SlimPajama-6B",
"instruction_dataset_repo": "your_data_path/mrqa-workshop_mrqa"
```
> 调整并行参数: 根据您的 GPU 数量修改梯度累积步数，确保： batch_size_per_device * device_count * gradient_accumulation_steps == total_batch_size

### 继续预训练(Pretrain)

```bash
cd pretrain
# 处理一次数据集即可
python pre_prepare_data.py --work_dir '../experiment/sac_experiment'
# 训练模型
python ./pre_trainer.py --work_dir '../experiment/sac_experiment' --port 14572
# 模型评估
python ./pre_evaluator.py --work_dir '../experiment/sac_experiment' --batch_size 1
```

### 微调 (Fine-tuning)

```bash
cd sft
# 处理一次数据集即可
python instruction_prepare_data.py --work_dir '../experiment/sac_experiment'
# 微调训练
python ./instruction_trainer.py --work_dir '../experiment/sac_experiment' --port 14527 > train.log 2>&1 &
# 模型评估
python ./instruction_evaluator.py --work_dir '../experiment/sac_experiment' --batch_size 1
# 结果分析
python ../util/evaluate_iid.py --work_dir '../experiment/sac_experiment'
python ../util/evaluate_ood.py --work_dir '../experiment/sac_experiment'
```

### 其他实验 (ICAE、500xCompressor、EPL ...)
若需进行 500xCompressor 相关实验，请切换至对应分支，操作流程相同：
```bash
git checkout 500xCompressor
```

## 📊 Results



## 📌 Citation
如果你觉得这项工作对你有帮助，欢迎引用我们的论文：

```
@misc{liu2025autoencodingfreecontextcompressionllms,
      title={Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors}, 
      author={Xin Liu and Runsong Zhao and Pengcheng Huang and Xinyu Liu and Junyi Xiao and Chunyang Xiao and Tong Xiao and Shengxiang Gao and Zhengtao Yu and Jingbo Zhu},
      year={2025},
      eprint={2510.08907},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.08907}, 
}
```






