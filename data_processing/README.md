# Dataset Processing


## Description
This project provides tools to process and reformat various datasets for translation, question answering and summarization. The provided Python scripts leverage the Hugging Face `datasets` library for efficient data handling.

## Installation

To get started, clone this repository and install the required packages:

```
pip install datasets
```

## Modules Overview

### `processing_utils.py`

This module includes functions to load configurations, process and transform datasets into specific formats depending on the task like multiturn conversation processing, sentiment analysis, or machine translation.

#### Key Functions:

- `loading_config`: Load dataset configurations.
- `loading_dataset`: Load specific slices of datasets based on configuration.
- `eval_reformat_jsonl`: Reformat evaluation datasets into JSONL format.
- `train_reformat_json`: Reformat training datasets to include conversational context.
- `angmubench_reformat_jsonl`: Prepare datasets for language-specific translation tasks.

### `extract_question.py`

Contains methods to extract questions and answers from dataset entries. Supports a wide range of tasks:

- `extract_math_gsm8k`, `extract_math_metamathqa`: Extract mathematical problem statements and solutions.
- `extract_translation_fren`, `extract_translation_deen`, `etc.`: Datasets for different language pairs.
- `extract_qa_nq`, `extract_qa_tqa`: Extract questions for QA tasks.
- `extract_summ_cnndm`, `extract_summ_samsum`, `extract_summ_xsum`: Extract document and summary pairs.
- `extract_senti_imdb`: Analyze sentiment from movie reviews.

### `run_data_processing.py`
This script is designed to facilitate the command-line reformatting of datasets based on the specified mode—training, evaluation, or benchmark testing. It utilizes configurations set up in `processing_utils.py` and `extract_question.py` to handle specific data formats.




## Usage

Here's how to use the scripts to process a dataset:

```bash 
python run_data_processing.py --mode train --dataset_configs_path './data_processing/dataset_configs.json' --train_output_path './dataset/train/output.json'
```

Adjust the parameters like `--mode`, `--dataset_configs_path`, `--train_output_path`, `--eval_output_path`, `--train_data_num`, and `--eval_data_num` to fit your data needs.



### Example Command

- For train dataset,
```
python data_processing/run_data_processing.py --mode train --train_data_num 65000 --train_output_path ./dataset/train/wmt/wmt16_deen_65000.json
```

- For eval dataset,
```
python data_processing/run_data_processing.py --mode eval --eval_output_path ./dataset/eval/question.jsonl
```
This command processes the WMT16 German-English translation dataset using the specified configurations and reformats it for further use.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your enhancements.