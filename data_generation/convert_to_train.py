import json
import tiktoken
import argparse
import random

from typing import List

def count_token(encoder, token_dict, obj):
    for conv in obj["conversations"]:
        if conv['from']=="human":
            result = encoder.encode(conv['value'])
            token_dict['human'] += len(result)
        elif conv['from']=="gpt":
            result = encoder.encode(conv['value'])
            token_dict['gpt'] += len(result)

def load_and_transform_json(input_file: str):
    """
    Load JSON objects from a file and transform them into a specified format.
    
    Args:
    input_file (str): Path to the input file containing JSON objects, one per line.
    
    Returns:
    list: A list of dictionaries with transformed data.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    token_dict = {"human": 0, "gpt": 0}
    transformed_data = []
    try:
        with open(input_file, 'r') as file:
            for line in file:
                obj = json.loads(line.strip())
                #count_token(encoder, token_dict, obj)
                new_item = {
                    "id": obj['id'],
                    "conversations": [{"from": conv['from'], "value": conv['value']} for conv in obj['conversations']]
                }
                transformed_data.append(new_item)
        print(token_dict)
    except IOError as e:
        print(f"Error reading file {input_file}: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_file}: {e}")
    
    return transformed_data

def reformat_json(input_files: List[str],
                  output_file: str):
    """
    Process multiple JSON files and save the combined and transformed data into one output file.
    
    Args:
    input_files (list of str): List of paths to the input JSON files.
    output_file (str): Path to the output file to save transformed data.
    """
    combined_data = []
    for input_file in input_files:
        combined_data.extend(load_and_transform_json(input_file))
    # suffle the data
    random.seed(20)
    random.shuffle(combined_data)
    try:
        with open(output_file, 'w') as file:
            json.dump(combined_data, file, indent=4)
        print(f"JSON data has been reformatted and saved to {output_file}")
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")

# Example usage
input_files = ['dataset/train/wmt_selfdistill_gemma/wmt19zhen_1m_0.0.json', 
               'dataset/train/wmt_selfdistill_gemma/wmt19zhen_1m_0.3.json', 
               'dataset/train/wmt_selfdistill_gemma/wmt19zhen_1m_0.7.json', 
               'dataset/train/wmt_selfdistill_gemma/wmt19zhen_1m_1.0.json', 
                ]
output_path = 'dataset/train/wmt_selfdistill_gemma/wmt19zhen_1m_reformat_shuffle.json'

reformat_json(input_files, output_path)

# shuffle_answer_file(output_path)
