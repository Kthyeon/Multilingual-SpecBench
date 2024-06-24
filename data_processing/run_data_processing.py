from processing_utils import *
import argparse
import os

# Category-Dataset Dictionary is import as DEFAULT_TASK_CLUSTERS_ABBREV
# You can check the Dictionary in processing_utils.py
# or print('defualt task dict :', DEFAULT_TASK_CLUSTERS_ABBREV)
TRAIN_TASK_ABBREV = collections.OrderedDict([
    ('translation', [
        'wmt16_deen',
        'wmt14_fren',
        'wmt16_ruen'
        'paracrawl_enfr'
        'wmt16_deen',
        'wmt14_fren',
    ]), 
])

EVAL_TASK_ABBREV = collections.OrderedDict([
    ('translation', [
        'wmt16_deen',
        'wmt14_fren',
        'wmt16_ruen'
        'wmt14_fren',
    ]), 
])

BENCH_TASK_ABBREV = collections.OrderedDict([
    ('translation', [
        'wmt16_deen',
        #'wmt14_fren',
        #'wmt16_ruen',
        #'jparacrawl_v3',
        #'klue_sts',
        #'wmt19_zhen',
        #'paracrawl_enit',
        #'paracrawl_enbg',
    ]), 
])

if __name__ == "__main__":
    # Set the dataset_configs and output file paths
    parser = argparse.ArgumentParser(description="Reformat dataset to train or evaluation form.")
    parser.add_argument("--mode",
                        type=str,
                        help="train or eval. train as sharegpt fit, eval as specbench fit")
    parser.add_argument("--dataset_configs_path",
                        type=str,
                        default='./data_processing/dataset_configs.json',
                        help="Path to the dataset configs file")
    parser.add_argument("--train_output_path",
                        type=str,
                        default='./dataset/train/wmt/wmt16_deen_20.json',
                        help="Path to save the output JSON file")
    parser.add_argument("--eval_output_path",
                        type=str,
                        default='./dataset/eval/question.jsonl',
                        help="Path to save the output JSON file")
    parser.add_argument("--train_data_num",
                        type=int,
                        default=20,
                        help="Number of each data to evaluation")
    parser.add_argument("--eval_data_num",
                        type=int,
                        default=80,
                        help="Number of each data to evaluation")

    args = parser.parse_args()
    
    # make output directory (./dataset/train/wmt/ and ./dataset/train/wmt_selfdistill/ and ./dataset/eval/)
    os.makedirs(os.path.dirname('./dataset/train/wmt/'), exist_ok=True)
    os.makedirs(os.path.dirname('./dataset/train/wmt_selfdistill/'), exist_ok=True)
    os.makedirs(os.path.dirname('./dataset/eval/'), exist_ok=True)


    # Allocate each variable by train/eval mode
    if args.mode == 'train':
        TASK_DICT = TRAIN_TASK_ABBREV 
        data_num = args.train_data_num
        output_path = args.train_output_path
        reformat_func = train_reformat_json

    elif args.mode =='eval':
        TASK_DICT = EVAL_TASK_ABBREV 
        data_num = args.eval_data_num
        output_path = args.eval_output_path
        reformat_func = eval_reformat

    elif args.mode =='angmu_bench':
        TASK_DICT = BENCH_TASK_ABBREV
        data_num = 80
        output_path = './dataset/eval/angmu_question_english.jsonl'
        reformat_func = angmubench_reformat_jsonl
        args.mode = 'eval'

    # Call the function to reformat as evaluation form
    for category_name, abbrev_task_names in TASK_DICT.items():
        print('category :', category_name)

        for abbrev_name in abbrev_task_names:
            print('dataset : '+ args.mode + '-' + abbrev_name)

            config = loading_config(abbrev_name, args.dataset_configs_path)
            dataset = loading_dataset(config, args.mode, data_num)
            
            reformat_func(dataset, category_name, config, output_path)