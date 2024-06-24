import json
import os
import concurrent.futures
from typing import List, Dict

import openai
import tqdm

import argparse
import logging
import natsort
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_random_exponential,
# )

from fastchat.model.model_adapter import get_conversation_template

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8001/v1"

api_base_pool = []

# List models API
for i in range(10):
    openai.api_base = "http://localhost:800{}/v1".format(i)
    try:
        models = openai.Model.list()["data"][0]["id"]
        api_base_pool.append(openai.api_base)
    except:
        break

print("API base pool: ", api_base_pool)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--num_threads", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument("--chat", action="store_true")
args = parser.parse_args()

# Assuming the ShareGPT format
data = json.load(open(args.data_path, "r"))

def generate_data(id: str, 
                    messages: List[Dict], 
                    idx: int,
                    output_path: str,
                    max_tokens: int = 2048,
                    temperature: float = 0.3
                    ):
    """ 
    Generates data for a single conversation.
    """
    try:
        openai.api_base = api_base_pool[idx % len(api_base_pool)]
        model_name = openai.Model.list()["data"][0]["id"]

        if args.chat:
            process_chat_messages(id, messages, model_name, output_path, max_tokens, temperature)
        else:
            process_non_chat_messages(id, messages, model_name, output_path, max_tokens, temperature)
    except Exception as e:
        logging.error(f"Failed to generate data for ID {id}: {str(e)}")


def process_chat_messages(id: str, 
                          messages: List[Dict], 
                          model_name: str,
                          output_path: str,
                          max_tokens: int = 2048,
                          temperature: float = 0.3):
    """
    Processes chat messages by sending them to the model and handling the response.

    Args:
        id (str): Identifier for the conversation.
        messages (List[Dict]): List of message dictionaries.
        model_name (str): Name of the model used for generating responses.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Controls randomness in the generation process.

    Messages are expected to have 'from' and 'value' keys.
    """

    converted_messages = []
    output_messages = []

    if messages[0]["from"] == "gpt":
        append_system_message(messages, converted_messages, output_messages)

    for message in messages[::2]:
        if message["from"] != "human":
            return

        converted_messages.append({"role": "user", "content": message["value"]})
        try:
            process_conversation(model_name, 
                                 converted_messages, 
                                 output_messages, 
                                 message,
                                 max_tokens,
                                 temperature)
        except Exception as e:
            print(f"Conversation processing failed: {str(e)}")
            break
    
    write_output_if_not_empty(output_messages, id, output_path)

def append_system_message(messages: List[Dict], 
                          converted_messages: List[Dict], 
                          output_messages: List[Dict]):
    """Appends a system message to the conversation history.

    Args:
        messages (List[Dict]): Full list of messages, will modify by removing the first message.
        converted_messages (List[Dict]): List to append converted system messages to.
        output_messages (List[Dict]): List to append original system messages to for output.
    """

    converted_messages.append({"role": "system", "content": messages[0]["value"]})
    output_messages.append(messages[0])
    messages.pop(0)


def process_conversation(model_name: str, 
                         converted_messages: List[Dict], 
                         output_messages: List[Dict], 
                         message: Dict, 
                         max_tokens: int, 
                         temperature: float):
    """Processes a single conversation turn by generating a response from the model.

    Args:
        model_name (str): Name of the model to use for generating the chat completion.
        converted_messages (List[Dict]): History of conversation formatted for the model.
        output_messages (List[Dict]): Accumulated output messages including the model responses.
        message (Dict): The latest user message to process.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Controls randomness in the generation process.

    Uses the provided model to generate a response to the given message based on the conversation history.
    Appends the user message and the model's response to the output messages.
    """
    try:
        conv = get_conversation_template(model_name)
        prompt = conv.get_prompt()

        response = openai.ChatCompletion.create(
            prompt=prompt,
            model=model_name,
            messages=converted_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            ignore_eos=True,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            request_timeout=6000
        )

        response_content = response.choices[0]['message']['content'].strip()
        output_messages.extend([
            message,
            {"from": "gpt", "value": response_content}
        ])
        if response.choices[0]['finish_reason'] == "length":
            logging.info("Response terminated due to max length for ID {}".format(message.get('id', 'Unknown')))
            return
        converted_messages.append({"role": "assistant", "content": response_content})

    except Exception as e:
        logging.error(f"Error processing conversation for {model_name}: {str(e)}")


def write_output_if_not_empty(output_messages: List[Dict], 
                              id: str,
                              output_path: str):
    if output_messages:
        with open(output_path, "a") as f:
            f.write(json.dumps({"id": id, "conversations": output_messages}) + "\n")
    else:
        print("No messages to output.")



def process_non_chat_messages(id: str, 
                              messages: List[Dict], 
                              model_name: str,
                              output_path: str,
                              max_tokens: int, 
                              temperature: float):
    """Processes non-chat messages by sending them to a model and handling the response.

    Args:
        id (str): Identifier for the conversation or message batch.
        messages (List[Dict]): List of message dictionaries.
        model_name (str): Name of the model used for generating responses.
        max_tokens (int): Maximum number of tokens the model is allowed to generate.
        temperature (float): The randomness of the response generation.
        output_path (str): Path to the file where responses are saved.

    Processes messages that do not follow a chat format but still require model interaction.
    """
    try:
        conv = get_conversation_template(model_name)

        if messages and messages[0]["from"] == "system":
            conv.system_message = messages[0]["text"]
            messages = messages[1:]  # Remove the system message from the list

        if messages:
            # Append user message to the conversation template
            conv.append_message(conv.roles[0], messages[0]["value"])
            # Append a placeholder for the assistant's response
            conv.append_message(conv.roles[1], None)
        
        prompt = conv.get_prompt()
        response = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            ignore_eos=False,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            request_timeout=60000
        )
        if conv.name=='vicuna_v1.1':
            response_text = response.choices[0]['text'].strip().rstrip('</s>')
        else:
            response_text = response.choices[0]['text'].strip().rstrip('<eos>')  
        save_response(id, prompt, response_text, output_path)
    except Exception as e:
        logging.error(f"Non-chat message processing failed for ID {id}: {str(e)}")

def save_response(id: str, 
                  prompt: str, 
                  response_text: str, 
                  output_path: str):
    """Saves the prompt and response text to a file or other storage, including the ID of the conversation.

    Args:
        id (str): Identifier for the conversation or message batch.
        prompt (str): The prompt sent to the model.
        response_text (str): The text generated by the model in response to the prompt.
        output_path (str): File path where the conversation is to be saved.
    """
    output_messages = {"id": id,
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": response_text}
                ]}
    with open(output_path, "a") as f:
        f.write(json.dumps(output_messages) + "\n")

def reorg_answer_file(answer_file: str):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as f_in:
        for line in f_in:
            qid = json.loads(line)["id"]
            answers[qid] = line

    qids = natsorted(list(answers.keys()))
    with open(answer_file, "w") as f_out:
        for qid in qids:
            f_out.write(answers[qid])

# if output_path exists, count the number of lines and skip the first n data
start = 0
if os.path.exists(args.output_path):
    with open(args.output_path, "r") as f:
        start = len(f.readlines())
        print("Skip first {} data".format(start))

with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for idx, sample in enumerate(data[start:]):
            future = executor.submit(
                generate_data,
                sample["id"],
                sample["conversations"],
                idx,
                args.output_path,
                args.max_tokens,
                args.temperature
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()
print("save the file")
reorg_answer_file(args.output_path)