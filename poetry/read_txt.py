import re
import json

# 定义正则表达式
delimiter_pattern = r'\n\n[\u4e00-\u9fff]\u3000'  # 匹配两个换行符后跟任意中文字符和一个全角空格

# 读取文本文件
file_path = '宋词三百首全解.txt'  # 替换为你的文本文件路径
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# 使用正则表达式分割文本
split_content = re.split(delimiter_pattern, text)

result_list = []
for i, part in enumerate(split_content):
    if len(part) > 10:
        pfirst_line, rest_of_text = part.split('\n\n', 1)
        result_list.append({
            "title": pfirst_line,
            "content": rest_of_text
        })
    else:
        print(part)

# 将结果保存到 JSON 文件中
output_file_path = '宋词三百首全解result.json'
with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(result_list, json_file, ensure_ascii=False, indent=4)

print(f"结果已保存至 {output_file_path}")