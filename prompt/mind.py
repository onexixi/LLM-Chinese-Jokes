mind="""
逻辑思维
批判性思维
创造性思维
系统思维
分析性思维
直觉思维
发散思维
收敛思维
抽象思维
具象思维
辩证思维
归纳思维
演绎思维
类比思维
反向思维
横向思维
纵向思维
战略性思维
战术性思维
整体思维
局部思维
长期思维
短期思维
概率思维
假设思维
反事实思维
元认知思维
结构化思维
非结构化思维
科学思维
哲学思维
艺术思维
数学思维
工程思维
设计思维
经济思维
政治思维
生态思维
系统动力学思维
矛盾思维
辩证法思维
形而上学思维
实证主义思维
相对主义思维
二元思维
多元思维
跨学科思维
平行思维
立体思维
网络思维
"""

import yaml

def read_yaml_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

file_path = 'mind.yaml'
data = read_yaml_file(file_path)

for key, value in data.items():
    print(f"{key}:")
    print(f"  描述: {value.get('描述', '')}")
    steps = value.get('具体执行步骤')
    if steps is not None:
        print("  具体执行步骤:")
        for step in steps:
            print(f"    - {step}")