import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import requests
from openai import OpenAI


class LLMProcessor:
    def __init__(self, db_path, openai_base_url, openai_api_key, num_threads=5):
        self.db_path = db_path
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.num_threads = num_threads
        self.client = OpenAI(base_url="http://localhost:5000/v1", api_key="810001a0a02948d5bf640a98cb69f653")
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.stop_event = threading.Event()
        self.init_db()
        self.models_url = f"{openai_base_url}/models"

    def get_available_models(self):
        modle = ""
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}"
            }

            # 尝试获取模型列表

            response = requests.get(f"{self.openai_base_url}/models", headers=headers)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and isinstance(data['data'], list):
                    modle = [model['id'] for model in data['data']][0]

            # 如果上面失败，尝试获取单个模型
            response = requests.get(f"{self.openai_base_url}/model", headers=headers)
            if response.status_code == 200:
                data = response.json()
                if 'id' in data:
                    modle = data['id']
            # 如果所有尝试都失败，返回空列表
            print("Failed to fetch models from all available endpoints.")
            return modle
        except Exception as e:
            print(f"Error fetching models: {str(e)}")
            return modle

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_execution_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    model TEXT NOT NULL,
                    result TEXT,
                    execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT
                )
            ''')
            conn.commit()

    def execute_llm_operation(self, prompt, model="defult"):
        try:
            client = self.client
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            result = completion.choices[0].message.content
            # print(f"Prompt: {prompt}\nResult: {result}\n")
            return result, "success"
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {str(e)}")
            return f"Error: {str(e)}", "error"

    def insert_result_to_db(self, prompt, model, result, status):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO llm_execution_results (prompt, model, result, status)
                    VALUES (?, ?, ?, ?)
                ''', (prompt, model, result, status))
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    def process_llm_request(self, prompt, able_model):
        if self.stop_event.is_set():
            return
        result, status = self.execute_llm_operation(prompt, able_model)
        self.insert_result_to_db(prompt, able_model, result, status)
        return result

    def add_request(self, prompt, model='defult'):
        if not self.stop_event.is_set():
            self.executor.submit(self.process_llm_request, prompt, model)

    def wait_for_completion(self):
        self.executor.shutdown(wait=True)

    def stop(self):
        self.stop_event.set()
        self.executor.shutdown(wait=False)
        print("Processor stopped.")


# 使用示例
if __name__ == "__main__":
    db_path = "llm_results.db"  # SQLite数据库文件路径
    openai_base_url = "http://localhost:5000/v1"
    openai_api_key = "810001a0a02948d5bf640a98cb69f653"
    client = OpenAI(base_url="http://localhost:5000/v1", api_key="810001a0a02948d5bf640a98cb69f653")
    processor = LLMProcessor(db_path, openai_base_url, openai_api_key, num_threads=5)
    available_models = processor.get_available_models()
    print(available_models)

    # 添加一些请求
    prompts = [
        "写一个笑话",
        "1+1+1",
        "你是谁",
        "什么微积分",
        "描述下黑动"
    ]

    for prompt in prompts:
        processor.add_request(prompt, model=available_models)

    # 等待所有请求完成
    processor.wait_for_completion()

    # 停止处理器
    processor.stop()

    print("All requests have been processed.")
