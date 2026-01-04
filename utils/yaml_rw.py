import yaml
from pathlib import Path

def load_yaml(yaml_path):
    """Load YAML configuration file"""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, yaml_path):
    """Save data to YAML configuration file"""
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

