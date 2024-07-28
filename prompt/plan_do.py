import random
import json
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.utils.json import _custom_parser
from openai import OpenAI


def do_query_llm(prompt, max_retries=3, retry_delay=2):
    client = OpenAI(base_url="http://localhost:5000/v1", api_key="810001a0a02948d5bf640a98cb69f653")
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="Qwen/qwen1_4-7b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            result = completion.choices[0].message.content
            print(result)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"查询失败,正在重试 (尝试 {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"查询失败,已达到最大重试次数: {e}")
                return None


class AnalysisPlanModule:
    def __init__(self):
        self.current_plan = self.initial_analysis_plan()

    def initial_analysis_plan(self):
        return [
            {"name": "步骤 1: 问题定义和系统思维", "methods": ["系统思维", "分析性思维", "批判性思维"],
             "data_collection": []},
            {"name": "步骤 2: 数据收集和分析", "methods": ["实证主义方法", "分析性思维", "网络思维"],
             "data_collection": []},
            {"name": "步骤 3: 根本原因分析", "methods": ["逻辑思维", "归纳思维", "演绎思维"], "data_collection": []},
            {"name": "步骤 4: 创新解决方案生成", "methods": ["创造性思维", "发散思维", "横向思维"],
             "data_collection": []},
            {"name": "步骤 5: 方案评估和决策", "methods": ["收敛思维", "多标准决策分析思维", "博弈论思维"],
             "data_collection": []},
            {"name": "步骤 6: 实施规划", "methods": ["系统思维", "长期思维", "敏捷思维"], "data_collection": []},
            {"name": "步骤 7: 未来影响评估", "methods": ["场景思维", "情景规划思维", "反事实思维"],
             "data_collection": []},
            {"name": "步骤 8: 反思和优化", "methods": ["元认知", "开放性思维", "批判性思维"], "data_collection": []}
        ]


class LLMServiceModule:
    def __init__(self, num_threads=3):
        self.num_threads = num_threads
        self.query_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.threads = []
        self.start_worker_threads()

    def start_worker_threads(self):
        for _ in range(self.num_threads):
            thread = threading.Thread(target=self.worker, daemon=True)
            thread.start()
            self.threads.append(thread)

    def worker(self):
        while True:
            prompt = self.query_queue.get()
            if prompt is None:
                break
            result = do_query_llm(prompt)
            self.result_queue.put(result)
            self.query_queue.task_done()

    def query(self, prompt):
        self.query_queue.put(prompt)
        return self.result_queue.get()

    def shutdown(self):
        for _ in range(self.num_threads):
            self.query_queue.put(None)
        for thread in self.threads:
            thread.join()


class StepExecutor:
    def __init__(self, analysis_plan, llm):
        self.analysis_plan = analysis_plan
        self.llm = llm

    def execute_step(self, problem, step_index):
        current_step = self.analysis_plan.current_plan[step_index]
        execution_prompt = f"""
执行以下分析步骤:
问题: {problem}
步骤: {current_step['name']}
思维方法: {', '.join(current_step['methods'])}
需要收集的数据: {', '.join(current_step['data_collection'])}

请提供详细的执行计划,包括:
1. 具体的执行步骤
2. 每个步骤使用的思维方法
3. 每个步骤需要收集的具体数据项
4. 如何使用LLM评估和分析收集到的数据
5. 预期的输出或结果

请用JSON格式回复,示例:
{{
  "执行步骤": [
    {{
      "步骤描述": "步骤1描述",
      "使用的思维方法": ["方法1", "方法2"],
      "数据收集": [
        {{
          "数据项": "数据项1",
          "收集方法": "收集方法描述",
          "LLM评估方法": "使用LLM评估该数据的方法"
        }},
        {{
          "数据项": "数据项2",
          "收集方法": "收集方法描述",
          "LLM评估方法": "使用LLM评估该数据的方法"
        }}
      ],
      "预期输出": "预期输出描述"
    }},
    {{
      "步骤描述": "步骤2描述",
      "使用的思维方法": ["方法3", "方法4"],
      "数据收集": [
        {{
          "数据项": "数据项3",
          "收集方法": "收集方法描述",
          "LLM评估方法": "使用LLM评估该数据的方法"
        }}
      ],
      "预期输出": "预期输出描述"
    }}
  ]
}}

请确保返回有效的JSON格式。
"""
        execution_result = self.llm.query(execution_prompt)
        try:
            execution_plan = _custom_parser(execution_result)
        except json.JSONDecodeError:
            print("执行计划格式错误,返回原始响应")
            execution_plan = {"error": "JSON解析失败", "raw_response": execution_result}

        return {
            "step": current_step['name'],
            "execution_plan": execution_plan
        }

    def optimize_plan(self, problem, current_step, result):
        optimization_prompt = f"""
基于以下信息,请提供优化当前分析计划的建议:
问题: {problem}
当前步骤: {current_step['name']}
当前分析计划: {json.dumps(self.analysis_plan.current_plan, ensure_ascii=False)}
执行结果: {json.dumps(result, ensure_ascii=False)}

请提供优化建议,格式为JSON,包含以下字段:
1. "add_steps": 要添加的步骤列表,每个步骤包含 "name", "methods", "data_collection" 字段
2. "remove_steps": 要删除的步骤名称列表
3. "modify_steps": 要修改的步骤,每个元素包含 "original" 和 "new" 两个字段,其中 "new" 包含完整的步骤信息

示例:
{{
    "add_steps": [
        {{
            "name": "新步骤1",
            "methods": ["方法1", "方法2"],
            "data_collection": ["数据项1", "数据项2"]
        }}
    ],
    "remove_steps": ["要删除的步骤名称"],
    "modify_steps": [
        {{
            "original": "原步骤名",
            "new": {{
                "name": "新步骤名",
                "methods": ["新方法1", "新方法2"],
                "data_collection": ["新数据项1", "新数据项2"]
            }}
        }}
    ]
}}

请确保返回有效的JSON格式。如果没有修改建议,请返回空列表。
"""
        optimization_response = self.llm.query(optimization_prompt)
        try:
            optimization_data = json.loads(optimization_response)
            self.apply_optimization(optimization_data)
        except json.JSONDecodeError:
            print("优化建议格式错误,跳过优化")

    def apply_optimization(self, optimization_data):
        for step in optimization_data.get("add_steps", []):
            self.analysis_plan.current_plan.append(step)

        self.analysis_plan.current_plan = [step for step in self.analysis_plan.current_plan
                                           if step["name"] not in optimization_data.get("remove_steps", [])]

        for modify in optimization_data.get("modify_steps", []):
            for i, step in enumerate(self.analysis_plan.current_plan):
                if step["name"] == modify["original"]:
                    self.analysis_plan.current_plan[i] = modify["new"]
                    break


class ImprovedAnalyzer:
    def __init__(self):
        self.analysis_plan = AnalysisPlanModule()
        self.llm = LLMServiceModule(num_threads=3)
        self.step_executor = StepExecutor(self.analysis_plan, self.llm)

    def analyze_problem(self, problem):
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for step_index in range(len(self.analysis_plan.current_plan)):
                future = executor.submit(self.step_executor.execute_step, problem, step_index)
                futures.append(future)

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                self.step_executor.optimize_plan(problem, self.analysis_plan.current_plan[len(results) - 1], result)

        return self.summarize_insights(results)

    def summarize_insights(self, results):
        insight_prompt = f"""
基于以下分析结果,总结关键洞察:
{json.dumps(results, ensure_ascii=False)}

请提供一个结构化的总结,包含以下部分:
1. 主要发现
2. 关键挑战
3. 推荐行动
4. 后续步骤
5. 数据洞察 (基于收集和分析的数据)
6. LLM贡献 (LLM在分析过程中的关键贡献)

请用JSON格式回复,示例:
{{
  "主要发现": ["发现1", "发现2"],
  "关键挑战": ["挑战1", "挑战2"],
  "推荐行动": ["行动1", "行动2"],
  "后续步骤": ["步骤1", "步骤2"],
  "数据洞察": ["洞察1", "洞察2"],
  "LLM贡献": ["贡献1", "贡献2"]
}}

请确保返回有效的JSON格式。
"""
        insights_response = self.llm.query(insight_prompt)
        try:
            insights = json.loads(insights_response)
            return insights
        except json.JSONDecodeError:
            print("洞察总结格式错误,返回原始响应")
            return {"error": "JSON解析失败", "raw_response": insights_response}

    def shutdown(self):
        self.llm.shutdown()


# 使用示例
problem = "如何提高公司的创新能力"
analyzer = ImprovedAnalyzer()
results = analyzer.analyze_problem(problem)

print("\n分析结果总结:")
print(json.dumps(results, ensure_ascii=False, indent=2))

analyzer.shutdown()