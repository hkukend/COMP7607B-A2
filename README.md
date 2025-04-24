# COMP7607B Assignment 2: Building and Fine-tuning MiniMind Language Model

**Course:** COMP7607B (2024-2025)
**Assignment:** 2 - Language Model Training Pipeline

**Important:** This codebase is exclusively for COMP7607B (2024-2025). Please do not upload your solutions or this codebase to any public platforms. All rights reserved.

---

## 1. Introduction: Training a Language Model from Scratch

### 1.1. What is MiniMind?

MiniMind is a compact yet powerful language model developed for this assignment. It follows a similar architecture to modern large language models but is designed to be trainable with limited computational resources. Through this assignment, you will build a complete training pipeline for MiniMind, from pretraining to specialized domain adaptation.

**Why is This Important?**

Understanding the full lifecycle of language model development is crucial for several reasons:

* **Foundational Knowledge:** Gain insights into how modern LLMs are created and optimized
* **Practical Experience:** Learn to handle real-world challenges in model training and fine-tuning
* **Specialized Applications:** Develop the skills to adapt language models for specific domains
* **Parameter Efficiency:** Understand techniques for efficiently updating models without retraining from scratch
* **RLHF:** Learn how to incorporate human feedback into model training
* **Research and Development:** Equip yourself with the skills needed for future research or industry roles in NLP

### 1.2. Language Model Training Pipeline

Training a language model involves several distinct phases, each building upon the previous one, in this assignment, we will focus on the translation task (English to Traditional Chinese). The pipeline consists of:

* **Pretraining:** Teaching the model to predict the next token in a sequence using a large bilingual corpus of text data
* **Supervised Fine-Tuning (SFT):** Refining the model's translation capabilities in general domains
* **Low-Rank Adaptation (LoRA):** Efficiently adapting the model's translation capabilities to specific domains (finance) without full fine-tuning
* **Direct Preference Optimization (DPO):** Further improving the model's translation capabilities using pairs of choose/reject responses

<div align="center">
  <img src="./assets/lora_diagram.png" height="300">
</div>
<p align="center">
  Figure 1. Illustration of LoRA adaptation technique (Source: Hugging Face)
</p>

### 1.3. Your Mission in This Assignment

In this assignment, you will implement a complete language model training pipeline for MiniMind. You will:

* **Collect Quality Data:** Gather appropriate datasets for each training phase
* **Complete the Codebase:** Implement the missing parts in the provided code
* **Pretrain:** Train the model on a large corpus of general text
* **Supervised Fine-Tune:** Refine the model's translation capabilities
* **Domain Adaptation:** Use LoRA to specialize the model for financial text translation
* **RLHF:** Implement DPO for further improvement of translation quality in the financial domain

---

## 2. Setting Up Your Environment

### 2.1. HKU GPU Farm: Recommended Computing Environment

The HKU GPU Farm provides the computational resources needed for training language models. Follow the provided [quickstart guide](https://www.cs.hku.hk/gpu-farm/quickstart) to set up your environment.

### 2.2. Local Setup: For Advanced Users

If you have access to a machine with a suitable GPU (NVIDIA with 16GB+ VRAM recommended), you can set up locally. Ensure you have the necessary drivers, CUDA, and cuDNN installed.

### 2.3. Environment Setup: Dependencies

**Python:** This codebase is tested with Python 3.11+.

**Virtual Environment (Recommended):** Use Anaconda to manage your project's dependencies:

```bash
conda create -n llm python=3.11
conda activate llm
pip install -r requirements.txt
```

---

## 3. Understanding the Codebase

### 3.1. Directory Structure

```bash
├── data/              # Data files for each training phase (all ready contains some demo data for testing)
│ ├── pretrain.jsonl   # Pretraining data
│ ├── sft.jsonl        # SFT data
│ ├── lora.jsonl       # LoRA data
│ ├── dpo.jsonl        # DPO data
│ └── upload.py        # Upload data to Hugging Face
├── model/
│ ├── tokenizer/       # Tokenizer files
│ ├── config.py        # Model configuration
│ ├── model.py         # MiniMind model architecture
│ └── lora.py          # LoRA implementation
├── dataset.py         # Dataset for each training phase
├── trainer.py         # Trainer for each training phase
├── train_pretrain.py  # Pretraining script
├── train_sft.py       # Supervised fine-tuning script
├── train_dpo.py       # Direct preference optimization script
├── train_lora.py      # LoRA training script
├── utils.py           # Utility functions
├── run.sh             # Shell script to run all training scripts
└── README.md          # Documentation
```

### 3.2. Key Components

* **MiniMind Model:** A transformer-based language model with configurable size
* **Dataset Classes:** Specialized dataset loaders for each training phase
* **Training Scripts:** Four separate training processes for different stages
* **LoRA Implementation:** Efficient fine-tuning method for domain adaptation

---

## 4. Your Tasks

### Task 1: Complete the Codebase

Look for sections marked with "`# Write Your Code Here`" in the provided files. You'll need to implement:

* The necessary parts in `model.py` to complete the model architecture
* The `LoRA` class in `model_lora.py` to support LoRA training
* DPO Loss function in `trainer.py`
* Data processing logic as needed

After you finished this task, please make sure that you pass the test cases in `test/` by running the following command:

```bash
pytest test/test_attention.py
pytest test/test_dpo.py
```

### Task 2: Data Collection

You'll need to find appropriate datasets for each training phase:

#### Pretraining Data

The `Pretraining` phase requires a large corpus of general text (english and chinese). Suggested sources:

* [The Pile](https://pile.eleuther.ai/) - A large, diverse dataset for language model pretraining
* [C4 (Colossal Clean Crawled Corpus)](https://huggingface.co/datasets/c4) - A clean web text corpus
* [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) - A reproduction of LLaMA training data

#### SFT Data

The `SFT` phase requires instruction-response pairs, here we focus on general translation.

* [WMT17 Chinese-English Dataset](https://www.statmt.org/wmt17/translation-task.html) - A standard benchmark dataset for English to Chinese translation, focusing on news text
* [OPUS-100 Dataset](https://huggingface.co/datasets/Helsinki-NLP/opus-100) - A multilingual corpus including Chinese-English pairs, sampled from the OPUS collection
* [United Nations Parallel Corpus](https://huggingface.co/datasets/Helsinki-NLP/un_pc) - A parallel corpus of UN documents, covering English and Chinese among other languages

#### LoRA Financial Data

The `LoRA` phase requires a financial parallel corpus. Suggested sources:

* [FFN: A Fine-grained Chinese-English Financial Domain Parallel Corpus](https://arxiv.org/abs/2406.18856) - A dataset containing 1,013 main texts and 809 titles from financial news articles, featuring manually corrected English-to-Chinese translations. Access may require contacting the authors.
* [Financial News Dataset for Text Mining](https://zenodo.org/record/5591908) - A comprehensive dataset of 60,473 bilingual (Chinese-English) news items from the Financial Times, spanning 2007 to 2021. Open access and available for direct download.
* [Financial datasets on Hugging Face](https://huggingface.co/datasets?search=financial) - Search for financial datasets

#### RLHF Data

The `DPO` phase requires preference pairs. You can use following method to generate preference pairs:

1. **Generate Translations:** Use your SFT model to generate translations for a set of sentences (here we focus on financial domain, can be same dataset from `LoRA` stage).
2. **Human Annotation:** Use LLM (ChatGPT, Grok, Sonnet, etc) to rank the translations based on quality.
3. **Create Preference Pairs:** Form pairs of translations based on the rankings.
4. **Data Format:** Ensure the preference pairs are in the correct format for training.
5. **Data Augmentation (Optional):** Consider using data augmentation techniques to increase the diversity of your preference pairs.

#### Data Format

Ensure your datasets are in the correct format for each training phase. You may need to preprocess the data to fit the expected input format.

* Pretraining: JSONL file with one sentence per line, please refer to the `data/pretrain.jsonl` for examples
* SFT: JSONL file with instruction-response pair per line, please refer to the `data/sft.jsonl` for examples
* LoRA: JSONL file with parallel sentences, please refer to the `data/lora.jsonl` for examples
* DPO: JSONL file with chose/reject pairs, please refer to the `data/dpo.jsonl` for examples

#### Upload Data to Hugging Face

Since the data may be too large to upload to Moodle, you can upload your data to Hugging Face and share the link with us.

Modify the `REPO_ID` and `TOKEN` in `data/upload.py` to your own repo id and token. Then run the script to upload your data to Hugging Face. **Remember to set the repo to public.**

```bash
python data/upload.py
```

After running the script, you can find the data in your Hugging Face account and the link will be saved in `data/hf_link.txt`. We will use the link to download the data for grading. **Again, remember to set the repo to public.**

### Task 3: Training Pipeline

Follow these steps to train your MiniMind model:

1. **Pretraining:** Run `train_pretrain.py` with your collected pretraining data
2. **Supervised Fine-Tuning:** Run `train_sft.py` with your SFT dataset
3. **Financial LoRA:** Run `train_lora.py` with your financial parallel corpus
4. **RLHF:** Run `train_dpo.py` with preference pairs

Tune the hyperparameters in each script to achieve the best performance.

---

## 5. Submission Guidelines

**If your student ID is 3030XXXXX, organize your submission as follows:**

```bash
3030XXXXX.zip
├── data/              # Data files for each training phase
│ └── hf_link.txt      # Link to Hugging Face dataset (your own repo)
├── model/
│ ├── tokenizer/       # Tokenizer files
│ ├── config.py        # Model configuration
│ ├── model.py         # MiniMind model architecture
│ └── lora.py          # LoRA implementation
├── test/              # Test set for evaluation
├── dataset.py         # Dataset for each training phase
├── evaluator.py       # Evaluation script
├── trainer.py         # Trainer for each training phase
├── train_pretrain.py  # Pretraining script
├── train_sft.py       # Supervised fine-tuning script
├── train_dpo.py       # Direct preference optimization script
├── train_lora.py      # LoRA training script
├── utils.py           # Utility functions
├── requirements.txt   # Python dependencies
└── run.sh             # Shell script to run all training scripts
```

## 6. Grading System

Your submission will be evaluated based on:

1. **Code Implementation (30%):** Correctness and quality of your code implementations
2. **Pre-training Perplexity (10%):** Perplexity of your pretrained model on the validation set
3. **SFT Translation Quality (20%):** COMET score for general text using your SFT weights
4. **Financial Translation (20%):** COMET score for financial text using your LoRA weights
5. **DPO Financial Translation (20%):** COMET score for financial text using your DPO weights

We will execute the `run.sh` script to rerun the training scripts with the your datasets and evaluate the performance on our test set.

**Grading Criterion:**

* Perplexity (Pre-train): < 15.0 on validation set
* COMET score (SFT): > 0.60
* COMET score (Lora): > 0.65
* COMET score (DPO): > 0.70

The following details explain how the grading system works, using the SFT task as an example: You will receive the full grade (100) for the task if your COMET score reaches 0.70. For COMET scores below 0.70, we will calculate your grade using a ratio. For instance, if you achieve a COMET score of 0.35, your score for the task will be calculated as 0.35 / 0.70, which equals 0.50, or 50% of the total grade.

**Important:**

1. Dataset Accessibility: Ensuring your datasets are accessible and well-organized is crucial. If they are not, you may only receive credit for the code implementation, regardless of model performance.
2. Translation Direction: The required translation direction is from English to Traditional Chinese. If your collected dataset is in Simplified Chinese, please convert it to Traditional Chinese. Note that model performance will be evaluated **only** on Traditional Chinese data, so this is essential.
3. Dataset Format: Please strictly follow the provided example format when preparing your dataset.

---

## 7. Resources and References

### 7.1. Technical References

* [Root Mean Square Layer Normalization (RMSNorm)](https://arxiv.org/abs/1910.07467)
* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
* [Attention is all you need](https://arxiv.org/abs/1706.03762)
* [GQA: A Simple and Effective Method for Multi-token Attention](https://arxiv.org/abs/2305.13245)
* [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
* [Supervised Fine-tuning](https://arxiv.org/abs/2109.01652)
* [COMET: A Neural Framework for MT Evaluation](https://arxiv.org/abs/2009.09025)

### 7.2. Tools

* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
* [COMET Metric](https://github.com/Unbabel/COMET)
* [Weights & Biases](https://wandb.ai/) (for experiment tracking)

---

## 8. Getting Help

* Post questions on the course forum
* Contact the teaching team via email

Good luck with your language model training journey!

---

## 9. Acknowledgements

* [MiniMind](https://github.com/jingyaogong/minimind)
