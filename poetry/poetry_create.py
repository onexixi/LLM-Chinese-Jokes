import json
from pathlib import Path
from openai import OpenAI
import retrying
import re
from poetry_prompt import art_poetry_prompt, art_translate_prompt
import concurrent.futures
import threading

# 配置常量
LLM_OPENAI_API_KEY = "810001a0a02948d5bf640a98cb69f653"
LLM_OPENAI_API_BASE = "http://localhost:5000/v1"
LLM_MODEL = "gpt-3.5-turbo-16k"

# 创建OpenAI客户端
client = OpenAI(base_url=LLM_OPENAI_API_BASE, api_key=LLM_OPENAI_API_KEY)

# 线程本地存储
thread_local = threading.local()


def get_client():
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI(base_url=LLM_OPENAI_API_BASE, api_key=LLM_OPENAI_API_KEY)
    return thread_local.client


@retrying.retry(wait_fixed=1000, stop_max_attempt_number=3)
def get_local_llm(user_input, prompt):
    client = get_client()
    completion = client.chat.completions.create(
        model="defut",  # 这个字段目前未使用
        messages=[{"role": "user", "content": prompt.format(user_input)}],
        temperature=0.7,
    )
    result=completion.choices[0].message.content
    print(f"输入：\n{prompt.format(user_input)}\n{result}")
    return result


def process_scene(scene, match_add_txt=''):
    match_add_txt += scene
    tr_txt = ''
    while not tr_txt:
        tr_txt = get_local_llm(match_add_txt, art_translate_prompt)
    cleaned_text = re.sub(r'\d+\.', '', tr_txt).replace('#', '').strip()
    return scene, cleaned_text


@retrying.retry(wait_fixed=1000, stop_max_attempt_number=3)
def get_result_json(user_input):
    result_ch_list, result_list = [], []
    try:
        result = get_local_llm(user_input, art_poetry_prompt)
        scenes = result.split('\n\n画面')

        for scene in scenes:
            try:
                ch_scene, en_scene = process_scene(scene)
                result_ch_list.append(ch_scene)
                result_list.append(en_scene)
            except Exception as e:
                print(f"翻译错误: {e}")

        final_tr_txt = get_local_llm(str(result), art_translate_prompt)
        result_ch_list.append(result)
        result_list.append(str(final_tr_txt))
    except Exception as e:
        print(f"解析画面错误: {e}")

    return result_list, result_ch_list


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_item(item):
    user_content = item.get("content", "")
    try:
        result_list, result_ch_list = get_result_json(user_content)
        return {
            "original": item,
            "result_list": result_list,
            "result_ch_list": result_ch_list
        }
    except Exception as e:
        print(f"处理项目时发生错误: {e}")
        return None


def main():
    input_file = Path("宋词三百首全解result.json")
    output_file = Path("optimized_result.json")

    input_data = read_json_file(input_file)

    # 使用线程池处理每个项目
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(process_item, input_data))

    # 过滤掉None结果（处理失败的项目）
    results = [r for r in results if r is not None]

    write_json_file(output_file, results)
    print(f"结果已保存到 {output_file}")


if __name__ == '__main__':
    main()