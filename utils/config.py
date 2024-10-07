import yaml

def load_config(yaml_path):
    with open(yaml_path, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)
    return cfg
