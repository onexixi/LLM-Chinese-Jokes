import json
import random
import time

from langchain_core.utils.json import _custom_parser

from llmwork.llm_job import LLMProcessor


def advanced_query_llm(prompt, llm_processor=None, num_iterations=3, num_outputs=3, max_retries=3):
    def single_query(p):
        # 这里应该是实际的LLM API调用
        # 为了演示，我们使用一个占位函数
        result = llm_processor.process_llm_request(p, "question_id", "")
        print(result)
        return result

    def criticize(response):
        critique_prompt = f"请严格批评以下回答，指出其中的问题和可以改进的地方：\n{response}"
        return single_query(critique_prompt)

    def improve(response, critique):
        improve_prompt = f"基于以下批评，改进原始回答：\n原始回答：{response}\n批评：{critique}"
        return single_query(improve_prompt)

    def generate_multiple(p):
        return [single_query(p) for _ in range(num_outputs)]

    def score_single(response):
        scoring_prompt = f"""
        请为以下回答进行评分。评分标准如下：
        1. 相关性（0-10分）：回答与问题的相关程度
        2. 准确性（0-10分）：回答中信息的准确程度
        3. 完整性（0-10分）：回答是否全面覆盖了问题的各个方面
        4. 创新性（0-10分）：回答中是否包含新颖的见解或方法
        5. 实用性（0-10分）：回答中的建议或解决方案是否可行

        请以JSON格式输出，格式如下：
        {{
            "relevance": 8,
            "accuracy": 7,
            "completeness": 9,
            "innovation": 6,
            "practicality": 8
        }}

        以下是需要评分的回答：
        {response}
        """

        for attempt in range(max_retries):
            try:
                scores_json = single_query(scoring_prompt)
                print(f"Raw scores JSON: {scores_json}")  # 调试信息
                scores_data = _custom_parser(scores_json)
                if all(key in scores_data for key in ['relevance', 'accuracy', 'completeness', 'innovation', 'practicality']):
                    return scores_data
                else:
                    raise ValueError("Scores data is incomplete")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt < max_retries - 1:
                    print(f"解析失败，正在进行第{attempt + 2}次尝试...")
                    time.sleep(1)  # 在重试之前稍作等待
                else:
                    print(f"解析失败{max_retries}次，使用随机分数")
                    return {'relevance': random.randint(1, 10),
                            'accuracy': random.randint(1, 10),
                            'completeness': random.randint(1, 10),
                            'innovation': random.randint(1, 10),
                            'practicality': random.randint(1, 10)}

    def score(responses):
        return [score_single(response) for response in responses]

    best_response = ""
    best_score = 0

    for _ in range(num_iterations):
        initial_response = single_query(prompt)
        critique = criticize(initial_response)
        improved_response = improve(initial_response, critique)
        multiple_responses = generate_multiple(improved_response)
        scores = score(multiple_responses)

        for i, score_dict in enumerate(scores):
            total_score = sum(score_dict.values())
            if total_score > best_score:
                best_score = total_score
                best_response = multiple_responses[i]

    return best_response, best_score

# 使用示例
if __name__ == '__main__':
    db_path = "llm_results.db"
    openai_base_url = "http://localhost:5000/v1"
    openai_api_key = "810001a0a02948d5bf640a98cb69f653"
    prompt = "请提出三个具有高度商业价值的 银发经济的商业模式"
    llm_processor = LLMProcessor(db_path, openai_base_url, openai_api_key)

    result, score = advanced_query_llm(prompt, llm_processor)
    print(f"最佳回答 (总得分: {score}):\n{result}")