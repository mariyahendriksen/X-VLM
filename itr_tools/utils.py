import json

def save_json(data, path):
    with open(path, 'w+') as f:
        json.dump(data, f)
    print('Saved data to ', path)

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    print('Loaded data from ', path)
    return data
