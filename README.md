# Multilingual Speculative Benchmark

## 💡 Overview

This repository is part of the research presented in our paper **"Towards Fast Multilingual LLM Inference: Speculative Decoding and Specialized Drafters"**. 

> **Abstract**: Large language models (LLMs) have transformed natural language processing, finding applications in various commercial domains. Despite their versatility, the deployment of these models often faces challenges, particularly with high inference times in multilingual settings. Our work introduces a novel training methodology involving speculative decoding, combined with language-specific drafter models that are trained through a targeted pretrain-and-finetune strategy. This approach significantly enhances inference speed and efficiency across different languages.


> **Authors**: **Euiin Yi*** (KAIST AI), **Taehyeon Kim*** (KAIST AI), **Hongseok Jeung** (KT), **Du-Seong Chang** (KT), **Se-Young Yun** (KAIST AI)
- **Euiin Yi** (KAIST AI) : euiin_mercyii@kaist.ac.kr
- **Taehyeon Kim** (KAIST AI) : potter32@kaist.ac.kr

The contributors marked with an asterisk (*) contributed equally to this work.


## 📑 Env settings
To set up your environment to use this repository, you can follow these steps using Conda. This will create a new environment called `multispec`:

```
conda create --name multispec python=3.9
conda activate multispec
conda install pip
pip install -r requirements.txt
```


## 💻 Repository Structure

- **Data_generation**: Scripts and utilities to generate self-distilled datasets for targeted LLMs, used to train specialized drafter models.
- **Data_processing**: Tools to process and download training and evaluation datasets for various tasks.
- **Evaluation**: Scripts to evaluate the performance improvements such as speedup and mean of accepted tokens when performing speculative inference.
- **LLM_judge**: Framework to assess the judgement score of various LLMs.
- **model**: Contains multiple speculative inference methods, including but not limited to Eagle, Medusa, Hydra, and Lookahead.
- **train**: Contains training functions and scripts used for training the specialized drafter models, with a focus on recommendations.

## 🚀 Getting Started

To get started with this repository, clone it using:
```bash
git clone https://github.com/<your-username>/Multilingual-Speculative-Benchmark.git
```

Follow the setup instructions in each directory to install necessary dependencies and set up the environment.

## 📌 Reference

This work builds upon the insights and frameworks provided by the [Speculative Benchmark repository](https://github.com/hemingkx/Spec-Bench). Our implementations and innovations are documented in the paper available in the repository.

## ✅ Citation

If you use our work or dataset, please cite:
```
Will be updated
```

## License

This project is licensed under the terms of the MIT license.

## Contributions

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

