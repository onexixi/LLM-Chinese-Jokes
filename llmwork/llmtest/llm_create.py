import json
from typing import List, Tuple, Dict

import numpy as np
from langchain_core.utils.json import _custom_parser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llmwork.llm_job import LLMProcessor


class BusinessIdeaProcessor:
    def __init__(self, llm_processor: LLMProcessor):
        self.llm_processor = llm_processor
        self.generated_ideas = []
        self.tfidf_vectorizer = TfidfVectorizer()

    def generate_business_ideas(self) -> List[Dict[str, str]]:
        prompt = self._get_generate_prompt()
        try:
            result = self.llm_processor.process_llm_request(prompt, "create", self.llm_processor.model)
            problems = [problem.split('. ', 1)[1].strip() for problem in result.split('\n') if problem.strip()]
            ideas = [self.generate_solution_for_problem(problem) for problem in problems]
            return self._filter_unique_ideas(ideas)
        except Exception as e:
            print(f"生成创意时发生错误: {e}")
            return []

    def generate_solution_for_problem(self, problem: str) -> Dict[str, str]:
        prompt = self._get_solution_prompt(problem)
        try:
            result = self.llm_processor.process_llm_request(prompt, "solve", self.llm_processor.model)
            solution_data = json.loads(result)
            return {
                "problem": problem,
                "solution": solution_data["解决方案"],
                "business_plan": solution_data["商业方案"]
            }
        except Exception as e:
            print(f"为问题 '{problem}' 生成解决方案时发生错误: {e}")
            return {"problem": problem, "solution": "", "business_plan": ""}

    def _filter_unique_ideas(self, ideas: List[Dict[str, str]]) -> List[Dict[str, str]]:
        unique_ideas = []
        for idea in ideas:
            if not self._is_similar_to_existing(idea):
                unique_ideas.append(idea)
                self.generated_ideas.append(idea)
        return unique_ideas

    def _is_similar_to_existing(self, new_idea: Dict[str, str], threshold: float = 0.8) -> bool:
        if not self.generated_ideas:
            return False

        new_text = f"{new_idea['problem']} {new_idea['solution']} {new_idea['business_plan']}"
        existing_texts = [f"{idea['problem']} {idea['solution']} {idea['business_plan']}" for idea in
                          self.generated_ideas]

        all_texts = existing_texts + [new_text]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

        return any(similarity > threshold for similarity in cosine_similarities)

    def generate_solution_for_problem(self, problem: str) -> Dict[str, str]:
        prompt = self._get_solution_prompt(problem)
        try:
            result = self.llm_processor.process_llm_request(prompt, "solve", self.llm_processor.model)
            solution_data = json.loads(result)
            return {
                "problem": problem,
                "solution": solution_data["解决方案"],
                "business_plan": solution_data["商业方案"]
            }
        except Exception as e:
            print(f"为问题 '{problem}' 生成解决方案时发生错误: {e}")
            return {"problem": problem, "solution": "", "business_plan": ""}

    def evaluate_ideas(self, ideas: List[Dict[str, str]]) -> List[Tuple[Dict[str, str], str, float]]:
        evaluated_ideas = []
        for idea in ideas:
            prompt = self._get_evaluate_prompt(idea)
            try:
                result = self.llm_processor.process_llm_request(prompt, "evaluate", self.llm_processor.model)
                evaluation_result = _custom_parser(result)
                data = json.loads(evaluation_result)
                if isinstance(data, dict) and '总分' in data:
                    score = data.get('总分', 0)
                    diversity_score = self._calculate_diversity_score(idea)
                    adjusted_score = (score + diversity_score) / 2  # 将多样性分数纳入总分
                evaluated_ideas.append(
                    (idea, json.dumps(evaluation_result, ensure_ascii=False, indent=4), adjusted_score))
            except Exception as e:
                print(f"评估创意 '{idea['problem']}' 时发生错误: {e}")
        return evaluated_ideas

    def refine_ideas(self, evaluated_ideas: List[Tuple[Dict[str, str], str, float]]) -> List[str]:
        refined_ideas = []
        for idea, evaluation, _ in evaluated_ideas:
            prompt = self._get_refine_prompt(idea, evaluation)
            try:
                result = self.llm_processor.process_llm_request(prompt, "refine", self.llm_processor.model)
                refined_ideas.append(result.strip())
            except Exception as e:
                print(f"优化创意 '{idea['problem']}' 时发生错误: {e}")
        return refined_ideas

    def _calculate_diversity_score(self, new_idea: Dict[str, str]) -> float:
        if not self.generated_ideas:
            return 10.0  # 如果是第一个创意，给予最高的多样性分数

        new_text = f"{new_idea['problem']} {new_idea['solution']} {new_idea['business_plan']}"
        existing_texts = [f"{idea['problem']} {idea['solution']} {idea['business_plan']}" for idea in
                          self.generated_ideas]

        all_texts = existing_texts + [new_text]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

        avg_similarity = np.mean(cosine_similarities)
        diversity_score = (1 - avg_similarity) * 10  # 将相似度转换为多样性分数
        return diversity_score

    def _get_generate_prompt(self) -> str:
        return """
        任务：生成10个独特的日常生活场景或问题，这些场景或问题可能启发新的商业创意。

        要求：
        1. 每个场景或问题应具体且简洁，不超过15个字。
        2. 场景或问题应涵盖不同的生活领域（如工作、娱乐、家庭等）。
        3. 重点关注可能被忽视但普遍存在的小问题或不便。

        输出格式：
        1. 场景/问题
        2. 场景/问题
        ...
        10. 场景/问题

        请直接列出10个场景或问题，无需其他解释。
        """

    def _get_solution_prompt(self, problem: str) -> str:
        return f"""
        问题：{problem}

        请为这个问题设计一个解决方案和相应的商业方案。输出应为JSON格式，包含以下字段：

        {{
            "解决方案": "简要描述技术或服务如何解决这个问题（50字以内）",
            "商业方案": "概述如何将这个解决方案商业化（100字以内，包括目标客户、收入模式等）"
        }}

        请确保输出是有效的JSON格式。
        """

    def _get_evaluate_prompt(self, idea: Dict[str, str]) -> str:
        return f"""
        任务：全面评估以下商业创意的潜力和可行性。

        问题：{idea['problem']}
        解决方案：{idea['solution']}
        商业方案：{idea['business_plan']}

        评估标准（每项1-10分）：
        1. 问题相关性：该创意解决的问题在现实生活中的普遍性和紧迫性。
        2. 创新程度：相比现有解决方案的新颖性和独特性。
        3. 市场潜力：潜在用户群体的规模和增长前景。
        4. 可行性：技术和资源要求，实现难度。
        5. 盈利能力：产生持续收入的潜力。
        6. 独特性：与其他常见解决方案的差异化程度。

        输出格式（JSON）：
        {{
            "问题相关性": {{ "分数": [分数], "解释": "[简要解释]" }},
            "创新程度": {{ "分数": [分数], "解释": "[简要解释]" }},
            "市场潜力": {{ "分数": [分数], "解释": "[简要解释]" }},
            "可行性": {{ "分数": [分数], "解释": "[简要解释]" }},
            "盈利能力": {{ "分数": [分数], "解释": "[简要解释]" }},
            "独特性": {{ "分数": [分数], "解释": "[简要解释]" }},
            "总分": [总分，满分60分],
            "总体评价": "[100字以内的综合分析，包括优势、劣势和建议]"
        }}

        请严格按照以上格式输出评估结果。
        """

    def _get_refine_prompt(self, idea: Dict[str, str], evaluation: str) -> str:
        return f"""
        任务：基于原始创意和评估结果，提出一个改进的商业创意。

        原始问题：{idea['problem']}
        原始解决方案：{idea['solution']}
        原始商业方案：{idea['business_plan']}

        评估结果：
        {evaluation}

        改进要求：
        1. 针对评估中指出的弱点进行改进。
        2. 保留并强化原创意的优势。
        3. 考虑如何扩大市场潜力和提高盈利能力。
        4. 确保改进后的创意具有可行性和创新性。

        输出格式：
        改进后的创意：[一句话简要描述，不超过20字]

        详细说明：
        [100字以内的详细描述，包括：
        - 创意的核心价值主张
        - 与原创意的主要区别
        - 预期能解决的问题
        - 潜在的目标用户群
        - 可能的盈利模式]

        请严格按照以上格式输出优化后的创意。
        """

    def process_ideas(self, num_iterations: int = 10) -> List[str]:
        all_refined_ideas = []
        for iteration in range(num_iterations):
            print(f"\n开始第 {iteration + 1} 轮创意生成和优化")

            initial_ideas = self.generate_business_ideas()
            if not initial_ideas:
                print("无法生成初始创意列表。跳过此轮。")
                continue

            evaluated_ideas = self.evaluate_ideas(initial_ideas)
            if not evaluated_ideas:
                print("无法评估创意。跳过此轮。")
                continue

            refined_ideas = self.refine_ideas(evaluated_ideas)
            if not refined_ideas:
                print("无法优化创意。跳过此轮。")
                continue

            all_refined_ideas.extend(refined_ideas)

        return all_refined_ideas


def main():
    db_path = "llm_results.db"
    openai_base_url = "http://localhost:5000/v1"
    openai_api_key = "810001a0a02948d5bf640a98cb69f653"

    llm_processor = LLMProcessor(db_path, openai_base_url, openai_api_key)
    business_idea_processor = BusinessIdeaProcessor(llm_processor)

    num_iterations = 10  # 设置循环次数
    final_ideas = business_idea_processor.process_ideas(num_iterations)

    if not final_ideas:
        print("未能生成任何有效的创意。程序终止。")
        return

    print("\n最终创意列表:")
    with open("idea_results.txt", "w", encoding="utf-8") as file:
        for i, idea in enumerate(final_ideas, 1):
            output = f"{i}. {idea}\n"
            print(output)
            file.write(output + "\n")

    print(f"\n所有创意已保存到 idea_results.txt 文件中。共生成 {len(final_ideas)} 个创意。")


if __name__ == "__main__":
    main()
