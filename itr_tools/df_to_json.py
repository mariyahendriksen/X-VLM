import argparse
import yaml
import pandas as pd
import os
from utils import save_json, csv_to_train_json_list, csv_to_test_dev_json_list


def main(args):
    dataset = args.dataset
    config_file = args.config_file
    dataset_split = args.dataset_split
    xvlm_root = args.xvlm_root
    is_small = args.is_small
    print(args)

    # load config file
    with open(config_file) as file:
        config_full = yaml.safe_load(file)
    # print(config_full)
    config = config_full[dataset]
    print('Loaded configuration: ', config)

    if is_small:
        csv_file = config['csv_file_small']
    else:
        csv_file = config['csv_file']

    # load df
    df = pd.read_csv(
        filepath_or_buffer=os.path.join(config['dataset_root'], csv_file),
        # filepath_or_buffer = '/Users/mhendriksen/Desktop/repositories/datasets/CUB_200_2011/cub_1_cap_per_img.csv',
        dtype=config['columns_dtypes'],
        index_col=0
        )
    df.head()
    print(f'Loaded df:{df.head()}\ndf shape: {df.shape}')

    df_subset = df[df['eval_status'] == dataset_split]
    print('Final df shape: ', df_subset.shape)

    # print("config['content_type']: ", config['content_type'])
    # print("config['content_type']['image']", str(config['content_type']['image']))
    
    if dataset_split == 'train':
        json_list = csv_to_train_json_list(
            df_subset, image_file_column=config['content_type']['image'], caption_column=config['content_type']['text']
            )
    else:
        json_list = csv_to_test_dev_json_list(
            df_subset, image_file_column=config['content_type']['image'], caption_column=config['content_type']['text']
        )
    
    if dataset_split == 'dev':
        dataset_split = 'val'
    
    if is_small:
        json_file = f'{dataset}_small_{dataset_split}.json'
    else:
        json_file = f'{dataset}_{dataset_split}.json'
    json_file_path = os.path.join(xvlm_root, json_file)
    save_json(json_list, json_file_path)
    
    print('Done!')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='cub',
                        choices=['cub', 'abo', 'fashion200k', 'coco', 'f30k'],
                        help='dataset type')
    parser.add_argument('--config_file', type=str, default='configs/data_conf.yaml',
                    help='Configuration file')
    parser.add_argument('--dataset_split', type=str,
                    default='train',
                    choices=['train', 'test', 'dev'],
                    help='train|test|dev dataset split')
    parser.add_argument('--is_small', type=bool, default=False,
                    help='Is it a small ds? Only for abo and fashion200k')
    parser.add_argument('--xvlm_root', type=str, default='/ivi/ilps/personal/mbiriuk/repro/X-VLM/finetune',
                    help='Xvlm fine tune root')
    args = parser.parse_args()
    main(args)