import json

def csv_to_train_json_list(dataf, image_file_column, caption_column):
    dicts_list = []
    for idx, row in dataf.iterrows():
        tmp_dict = {
        'caption': row[caption_column],
        'image': row[image_file_column],
        'image_id': idx
        }
        dicts_list.append(tmp_dict)
    assert len(dicts_list) == len(dataf)
    print(f'Got list with {len(dicts_list)} dicts')
    return dicts_list

def csv_to_test_dev_json_list(dataf, image_file_column, caption_column):
    dicts_list = []
    for _, row in dataf.iterrows():
        tmp_dict = {
        'image': row[image_file_column],
        'caption': [row[caption_column]],
        }
        dicts_list.append(tmp_dict)
    assert len(dicts_list) == len(dataf)
    print(f'Got list with {len(dicts_list)} dicts')
    return dicts_list 

def save_json(data, path):
    with open(path, 'w+') as f:
        json.dump(data, f)
    print('Saved data to ', path)

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    print('Loaded data from ', path)
    return data
