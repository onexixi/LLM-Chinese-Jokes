import json
import logging
import multiprocessing
import os
import random
import re
import time
from functools import wraps

import matplotlib.pyplot as plt
from langchain_core.utils.json import _custom_parser

from llmwork.llm_job import LLMProcessor


def retry_with_exponential_backoff(max_retries=3, initial_wait=1, exponential_base=2, jitter=0.1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            num_retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    num_retries += 1
                    if num_retries > max_retries:
                        raise e
                    wait_time = initial_wait * (exponential_base ** (num_retries - 1))
                    wait_time += random.uniform(-jitter * wait_time, jitter * wait_time)
                    print(f"重试 {num_retries}/{max_retries}，等待 {wait_time:.2f} 秒")
                    time.sleep(wait_time)

        return wrapper

    return decorator


class PromptOptimizationAgent:
    def __init__(self, db_path, openai_base_url, openai_api_key, initial_prompt, task_type, iterations=10,
                 max_iterations=20, min_iterations=5):
        self.db_path = db_path
        self.openai_base_url = openai_base_url
        self.openai_api_key = openai_api_key
        self.initial_prompt = initial_prompt
        self.task_type = task_type
        self.iterations = iterations
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.prompt_library = PromptLibrary()
        self.performance_history = []
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(f"PromptOptimizationAgent_{self.task_type}")
        logger.setLevel(logging.INFO)

        # 确保日志目录存在
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 使用 UTF-8 编码创建 FileHandler
        log_file = os.path.join(log_dir, f"optimization_{self.task_type}.log")
        handler = logging.FileHandler(log_file, encoding='utf-8')

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def optimize(self):
        llm_processor = LLMProcessor(self.db_path, self.openai_base_url, self.openai_api_key)
        try:
            best_prompt = self.initial_prompt
            best_score = 0
            no_improvement_count = 0

            for i in range(self.max_iterations):
                self.logger.info(f"开始第 {i + 1} 次迭代")
                current_prompt = self.initial_prompt if i == 0 else self.dynamic_optimize(llm_processor, best_prompt, i)
                criticism = self.criticize(llm_processor, current_prompt)
                improved_prompt = self.modify(llm_processor, current_prompt, criticism)

                performance = self.evaluate_prompt(llm_processor, improved_prompt)

                self.performance_history.append(performance)
                self.logger.info(f"性能评分：{performance}")


                self.logger.info(f"目前最高分：{best_score}")

                if i >= self.min_iterations:
                    if no_improvement_count >= 3:
                        self.logger.info("连续3次无改进，提前结束优化")
                        break
                    elif best_score >= 9.5:
                        self.logger.info("达到高分，提前结束优化")
                        break

            self.prompt_library.add_prompt(self.task_type, best_prompt, best_score)
            self.visualize_performance()
            return best_prompt
        except Exception as e:
            self.logger.error(f"优化过程中发生错误：{e}")
            return self.initial_prompt

    def dynamic_optimize(self, llm_processor, prompt, iteration):
        try:
            related_prompts = self.prompt_library.get_related_prompts(self.task_type)

            optimization_prompt = f"""
            我们正在进行第 {iteration} 次迭代，优化以下提示词：

            当前提示词：{prompt}

            任务类型：{self.task_type}
            历史性能：{self.performance_history}

            相关任务的高性能提示词：
            {related_prompts}

            基于以上信息，请提供一个改进后的提示词版本。
            考虑以下几个方面：
            1. 清晰度和具体性
            2. 与任务类型的相关性
            3. 鼓励更详细或创造性的回答
            4. 解决历史性能中显示的任何弱点
            5. 从相关任务的高性能提示词中学习有效的模式

            请提供您优化后的提示词：
            """

            optimized_prompt = llm_processor.process_llm_request(optimization_prompt, f"optimize_{iteration}",
                                                                 llm_processor.model)
            self.logger.info(f"优化后的提示词：{optimized_prompt}")
            return optimized_prompt
        except Exception as e:
            self.logger.error(f"动态优化过程中发生错误：{e}")
            return prompt

    def clean_prompt(self, llm_processor, prompt):
        try:
            cleaning_prompt = f"""
            请仔细阅读以下提示词，并删除所有不直接相关于主题 "{self.task_type}" 的内容。
            保留所有与主题直接相关的关键信息和指令。
            删除任何可能导致LLM偏离主题或产生不相关输出的内容。
            确保清理后的提示词简洁明了，直接针对任务要求。

            原始提示词：
            {prompt}

            请提供清理后的提示词：
            """

            cleaned_prompt = llm_processor.process_llm_request(cleaning_prompt, f"clean_prompt_{self.task_type}",
                                                               llm_processor.model)

            cleaned_prompt = re.sub(r'^["\']|["\']$', '', cleaned_prompt.strip())
            cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt)

            return cleaned_prompt
        except Exception as e:
            self.logger.error(f"清理提示词过程中发生错误：{e}")
            return prompt

    def criticize(self, llm_processor, prompt):
        try:
            prompt_list = [
                "###Instruction### 作为一位经验丰富的提示词工程师，你的任务是对给定的提示词进行严格的批评分析。你必须指出其中的弱点和可能的改进之处。",
                "你将扮演一个挑剔的AI专家角色。详细分析提供的提示词，并指出其中的问题和不足。",
                "逐步思考并分析以下提示词的优缺点。确保你的回答没有偏见，避免刻板印象。",
                "以自然、类人化的方式回答问题：这个提示词有什么问题？它如何才能更好？",
                "###Instruction### 你的任务是批评性地分析以下提示词。###Question### 这个提示词的主要缺陷是什么？如何改进？",
                "详细地为我写一篇关于给定提示词缺陷的分析，并添加所有必要的信息。",
                "Capacity and Role：你是一位严格的提示词审查员。Insight：好的提示词应该清晰、具体、相关。Statement：分析并批评给定的提示词。Personality：直接且严厉。",
                "你将受到惩罚如果不能提供深入、有建设性的批评。请全面分析提示词的结构、内容和有效性。",
                """使用以下短语："你的任务是"和"你必须"。""",
                """不用加入诸如“请”，“如果你不介意”，“谢谢”等短语，直截了当地表达观点。。""",
                """把目标受众融入到提示中，例如，受众是该领域的专家。""",
                """以我11岁的样子向我解释。""",
                """以我在【领域】的初学者身份向我解释。""",
                """用简单的英语写出【文章/文本/段落】，就像你在向一个5岁的孩子解释一样。""",
                """添加”生成得好我会你$xxx小费！"的句子。""",
                """使用示例驱动的提示（使用少量示例进行提示）。""",
                """在格式化提示时，以”###Instruction###’"开头，后面跟着如果相关的话要么是"###。""",
                """Example###"要么是"###Question###"。然后给出你的内容。用一个或多个换行来分隔指令、示例、问题、背景和输入数据。""",
                """以我11岁的样子向我解释。""",
                """分隔指令、示例、问题、背景和输入数据。""",
                """使用以下短语："你的任务是"和"你必须"。""",
                """使用以下短语："你将受到惩罚"。""",
                """在提示中使用短语"以自然、类人化的方式回答问题"。""",
                """使用引导词，比如写"逐步思考"。""",
                """在提示中添加以下短语："确保你的回答没有偏见，避免刻板印象"。""",
                """允许模型通过向您提问，直到他获得足够的信息来提供所需的输出，从而引出精确的细节和要求。（例如，“从现在开始，我希望你问我问题，以便...”）。""",
                """要询问有关特定主题、想法或任何信息并且想要测试自己的理解，您可以使用以下短语：“教我[任何定理/主题/规则名称]，并在最后包含一个测试，但不要给我答案，当我回答时告诉我答案是否正确”。""",
                """为大模型分配一个角色。""",
                """使用分隔符。""",
                """在提示中重复特定的单词或短语多次。""",
                """结合思维链(CoT)和少样本提示。""",
                """使用输出引导语，这涉及到以期望的输出开头来结束你的提示。利用输出引导语，通过以预期响应的开头来结束你的提示。""",
                """要写一篇详细的论文/文本/段落/文章或任何类型的文本，可以使用以下指示："详细地为我写一篇关于[主题]的[论文/文本/段落]，并添加所有必要的信息"。""",
                """要更正/修改特定的文本而不改变其风格，可以使用以下指示："尝试修改用户发送的每个段落。你只需改进用户的语法和用词，确保其读起来自然。不要改变写作风格，比如将正式段落变得随意"。""",
                """清楚地陈述模型必须遵循的要求，以关键词、规定、提示或指令的形式提供内容。""",
                """要写任何类型的文本，如论文或段落，使其类似于提供的样本，包括以下指示："请根据提供的段落[/标题/文本/论文/答案]使用相同的表达"""
            ]

            selected_prompts = random.sample(prompt_list, 3)  # 随机选择3个提示

            criticism_prompt = f"""
            {' '.join(selected_prompts)}

            待批评的提示词：{prompt}

            任务类型：{self.task_type}

            请提供详细的批评，重点关注：
            1. 清晰度和具体性
            2. 与任务的相关性
            3. 结构和逻辑
            4. 创新性和启发性
            5. 可操作性

            你的批评：
            """

            criticism = llm_processor.process_llm_request(criticism_prompt, "criticize", llm_processor.model)
            print(f"批评：{criticism}")
            return criticism
        except Exception as e:
            print(f"批评过程中发生错误：{e}")
            return "没有具体的批评可用。"

    def modify(self, llm_processor, prompt, criticism):
        try:
            modification_prompt = f"""
            基于以下批评，请修改并改进给定的提示词：

            原始提示词：{prompt}

            批评：{criticism}

            任务类型：{self.task_type}

            请提供一个改进后的提示词版本，解决批评中指出的问题：
            """

            modified_prompt = llm_processor.process_llm_request(modification_prompt, "modify",
                                                                llm_processor.model)
            print(f"修改后的提示词：{modified_prompt}")
            return modified_prompt
        except Exception as e:
            print(f"修改过程中发生错误：{e}")
            return prompt

    @retry_with_exponential_backoff(max_retries=3, initial_wait=1, exponential_base=2, jitter=0.1)
    def evaluate_prompt(self, llm_processor, prompt):
        try:
            evaluation_prompt = f"""
            请对以下提示词进行严格且细致的评估，考虑其对任务类型"{self.task_type}"的有效性。评分范围为0到10分，请确保评分能够体现出不同提示词之间的明显差异。

            待评估的提示词：{prompt}

            请按照以下严格的评分标准进行评估：

            1. 清晰度和具体性（0-2分）：
               - 提示词是否清晰易懂？
               - 是否提供了足够的细节和具体指导？

            2. 相关性和针对性（0-2分）：
               - 提示词与任务类型的相关程度如何？
               - 是否针对特定任务提供了适当的引导？

            3. 创新性和启发性（0-2分）：
               - 提示词是否鼓励创新思考？
               - 是否能启发深入的探讨和分析？

            4. 结构和逻辑（0-2分）：
               - 提示词的结构是否合理？
               - 各部分之间是否有逻辑联系？

            5. 可操作性和可回答性（0-2分）：
               - 基于此提示词是否容易生成回答？
               - 提示词是否过于宽泛或过于狭窄？

            请使用JSON格式提供您的详细评估：
            {{
                "total_score": <总分，精确到小数点后一位>,
                "clarity_score": <清晰度和具体性得分>,
                "clarity_comment": "<对清晰度评分的简要解释>",
                "relevance_score": <相关性和针对性得分>,
                "relevance_comment": "<对相关性评分的简要解释>",
                "innovation_score": <创新性和启发性得分>,
                "innovation_comment": "<对创新性评分的简要解释>",
                "structure_score": <结构和逻辑得分>,
                "structure_comment": "<对结构评分的简要解释>",
                "actionability_score": <可操作性和可回答性得分>,
                "actionability_comment": "<对可操作性评分的简要解释>",
                "strengths": [<提示词的主要优点，列出至少2-3点>],
                "weaknesses": [<提示词的主要缺点，列出至少2-3点>],
                "improvement_suggestions": "<具体的改进建议>"
            }}

            请确保您的评分是严格且有区分度的，充分反映提示词的质量差异。同时，请提供详细的解释和建设性的反馈。
            """

            evaluation_json = llm_processor.process_llm_request(evaluation_prompt, "evaluate",
                                                                llm_processor.model)
            self.logger.info(f"原始评估响应：{evaluation_json}")

            try:
                evaluation = json.loads(evaluation_json)
            except json.JSONDecodeError:
                self.logger.warning("JSON解析失败，尝试使用自定义解析器")
                evaluation = _custom_parser(evaluation_json)

            self.logger.info(f"解析后的评估：{evaluation}")

            if isinstance(evaluation, dict) and 'total_score' in evaluation:
                return evaluation['total_score']
            else:
                self.logger.error("评估结果中不包含'total_score'键")
                raise ValueError("评估结果格式不正确")
        except Exception as e:
            self.logger.error(f"提示词评估过程中发生错误：{e}")
            raise

    def visualize_performance(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.performance_history) + 1), self.performance_history)
        plt.title(f"Prompt Optimization Performance for {self.task_type}")
        plt.xlabel("Iteration")
        plt.ylabel("Performance Score")
        plt.savefig(f"performance_{self.task_type}.png")
        plt.close()


class PromptLibrary:
    def __init__(self):
        self.prompts = {}

    def add_prompt(self, task_type, prompt, performance_score):
        if task_type not in self.prompts:
            self.prompts[task_type] = []
        self.prompts[task_type].append((prompt, performance_score))

    def get_best_prompt(self, task_type):
        if task_type in self.prompts:
            return max(self.prompts[task_type], key=lambda x: x[1])[0]
        return None

    def get_related_prompts(self, task_type, num_prompts=3):
        related_prompts = []
        for t, prompts in self.prompts.items():
            if t != task_type:
                related_prompts.extend(prompts)
        related_prompts.sort(key=lambda x: x[1], reverse=True)
        return related_prompts[:num_prompts]


def optimize_prompt_parallel(args):
    db_path, openai_base_url, openai_api_key, initial_prompt, task_type, iterations = args
    agent = PromptOptimizationAgent(db_path, openai_base_url, openai_api_key, initial_prompt, task_type, iterations)
    return agent.optimize()


if __name__ == "__main__":
    db_path = "llm_results.db"
    openai_base_url = "http://localhost:5000/v1"
    openai_api_key = "810001a0a02948d5bf640a98cb69f653"
    llm_processor = LLMProcessor(db_path, openai_base_url, openai_api_key)

    # 定义多个任务和初始提示词
    tasks = [
        ("AI_explanation", "未来银发经济有关的商机"),
        ("Data_analysis", "分析未来银发经济趋势"),
        ("Creative_writing", "写一个银发经济有关的商机"),
        ("Market_research", "调研年轻人'搭子文化'的商业潜力"),
        ("Product_ideation", "为'搭子文化'设计一款社交App"),
        ("Trend_analysis", "分析短视频平台上年轻人的消费趋势"),
        ("Business_model", "设计一个基于共享经济的年轻人出行解决方案"),
        ("Content_strategy", "为一个面向Z世代的新媒体平台制定内容策略"),
        ("Service_design", "设计一个满足年轻人个性化需求的订阅式服务"),
        ("Marketing_campaign", "为一个新兴的国潮品牌策划针对年轻人的营销活动"),
        ("User_experience", "优化一个面向年轻用户的在线教育平台的用户体验"),
        ("Sustainability", "探索环保理念如何融入年轻人的生活方式并创造商机"),
        ("Gaming_industry", "分析游戏产业中针对年轻人的新兴商业模式"),
        ("Health_wellness", "为注重身心健康的年轻群体开发一款智能健康管理产品"),
        ("Financial_services", "设计一款帮助年轻人理财和投资的金融科技产品")
    ]

    with multiprocessing.Pool() as pool:
        optimized_prompts = pool.map(optimize_prompt_parallel,
                                     [(db_path, openai_base_url, openai_api_key, prompt, task_type, 10) for
                                      task_type, prompt in tasks])

    for (task_type, _), optimized_prompt in zip(tasks, optimized_prompts):
        print(f"Task: {task_type}")
        print(f"Optimized prompt: {optimized_prompt}")
        print()
