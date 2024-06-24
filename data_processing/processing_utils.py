import json
import collections

from datasets import load_dataset 
from extract_question import Extract_question_answer



DEFAULT_TASK_CLUSTERS_ABBREV = collections.OrderedDict([
    ('multiturn', [
        'mt_bench',
    ]),
    ('summarization', [
        'cnn_dailymail',
        'samsum',
        'xsum'
    ]),
    ('qa', [
        'natural_questions',
        'trivia_qa',
        'arc_easy',
        'arc_challenge',
    ]), 
    ('sentiment', [
        'imdb',
    ]), 
    ('translation', [
        'wmt14_fren',
        'wmt16_deen',
        'wmt16_csen',
        'wmt16_fien',
        'wmt16_roen',
        'wmt16_ruen',
        'wmt16_tren',
        'paracrawl_enes',
        'paracrawl_enfr',
    ]), 
    ('math_reasoning', [
        'gsm8k',
        'metamathqa',
    ])
])

# loading dataset's config
def loading_config(data_name: str, 
                   dataset_configs_path: str):

    dataset_configs = json.load(open(dataset_configs_path, "r"))
    config = dataset_configs[data_name]

    return config

# loading dataset from hugging_face repo
def loading_dataset(config: dict,
                    mode: str,
                    data_num: int):
    # allocate split mode by train/eval
    # sometimes config[mode] is null since there is no split eval dataset,
    # then using the last data of train dataset
    if config[mode] != None:
        split_data = config[mode] + f'[:{data_num}]'
    else: 
        split_data = config["train"] + f'[-{data_num}:]'

    # loading the proper dataset
    if config["subset"] != None :
        dataset = load_dataset(config["data_path"], config["subset"], split=split_data)
    else:
        dataset = load_dataset(config["data_path"], split=split_data)

    return dataset

def eval_reformat_jsonl(dataset: list,
                   category: str,
                   config: dict,
                   output_file: str):

    transformed_data = []
    id_num = 1 

    for obj in dataset:
        extract_qa = getattr(Extract_question_answer, config['qa_extraction_fn'])
        question, answer = extract_qa(obj)
        new_item = {
            "question_id": f"{config['abbrev_name']}_{id_num}",
            "category": category,
            "turns": [question],
        }
        if answer is not None:
            new_item["reference"] = [answer]
        transformed_data.append(new_item)
        id_num += 1

    # Save the transformed data to the output file
    with open(output_file, 'a', encoding= "utf-8") as file:
        for i in transformed_data:
            file.write(json.dumps(i) + "\n")

    print("JSON data has been reformatted and saved to", output_file)


def eval_reformat_jsonl_mt(dataset: list,
                   category: str,
                   config: dict,
                   output_file: str):
    transformed_data = []
    id_num = 1 
    for obj in dataset:
        extract_qa = getattr(Extract_question_answer, config['qa_extraction_fn'])
        mt_category, question, answer = extract_qa(obj)
        new_item = {
            "question_id": f"{config['abbrev_name']}_{id_num}",
            "category": f"mt_{mt_category}",
            "turns": question,
        }
        if answer is not None:
            new_item["reference"] = answer
        transformed_data.append(new_item)
        id_num += 1

    # Save the transformed data to the output file
    with open(output_file, 'a', encoding= "utf-8") as file:
        for i in transformed_data:
            file.write(json.dumps(i) + "\n")

    print("JSON data has been reformatted and saved to", output_file)

def eval_reformat(dataset: list,
                   category_name: str,
                   config: dict,
                   output_file: str):
    if category_name == 'multiturn':
        eval_reformat_jsonl_mt(dataset, category_name, config, output_file)
    else:
        eval_reformat_jsonl(dataset, category_name, config, output_file)

def train_reformat_json(dataset: list,
                   category: str,
                   config: dict,
                   output_file: str):

    transformed_data = []
    id_num = 1 

    for obj in dataset:
        extract_qa = getattr(Extract_question_answer, config['qa_extraction_fn'])
        question, answer = extract_qa(obj)

        new_item = {
            "id": f"{config['abbrev_name']}_{id_num}",
            "conversations": []
        }
        new_item['conversations'].append(
                {
                    "from": "human",
                    "value": question
                })
        new_item['conversations'].append(
                {
                    "from": "gpt",
                    "value": answer
                })
        transformed_data.append(new_item)
        id_num += 1
        

    # Save the transformed data to the output file
    with open(output_file, 'a') as file:
        json.dump(transformed_data, file, indent=4)

    print("JSON data has been reformatted and saved to", output_file)

def angmubench_reformat_jsonl(dataset: list,
                   category: str,
                   config: dict,
                   output_file: str):

    transformed_data = []

    for language in ['German', 'French', 'Russian', 'Japanese', 'Korean', 
                     'Chinese', 'Italian', 'Bulgarian']:
        id_num = 1 
        for obj in dataset:
            #extract_qa = getattr(Extract_question_answer, config['qa_extraction_fn'])
            #question, answer = extract_qa(obj)
            question = obj['translation']['en']
            new_item = {
                "question_id": f"{config['abbrev_name']}_{id_num}",
                "category": category,
                "turns": [f'Translate English to {language}:'+ question],
            }
            transformed_data.append(new_item)
            id_num += 1
        print(language, 'finish!')
        # Save the transformed data to the output file
    with open(output_file, 'a', encoding= "utf-8") as file:
        for i in transformed_data:
            file.write(json.dumps(i) + "\n")

    print("JSON data has been reformatted and saved to", output_file)