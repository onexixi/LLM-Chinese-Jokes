import yaml

def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None

def print_structure(data, indent=0):
    if isinstance(data, dict):
        for key, value in data.items():
            print("  " * indent + f"{key}:")
            print_structure(value, indent + 1)
    elif isinstance(data, list):
        for item in data:
            print_structure(item, indent)
    else:
        print("  " * indent + str(data))

# 使用示例
file_path = 'zh_CN.yaml'  # 替换为您的YAML文件路径
yaml_data = read_yaml(file_path)

if yaml_data:
    print_structure(yaml_data)