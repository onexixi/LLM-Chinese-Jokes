import random
from datetime import datetime
from typing import List, Tuple, Dict
from llmwork.llm_job import LLMProcessor
from langchain_core.utils.json import _custom_parser
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import concurrent.futures

class BusinessIdeaProcessor:
    def __init__(self, llm_processor: LLMProcessor):
        self.llm_processor = llm_processor
        self.generated_ideas = []
        self.tfidf_vectorizer = TfidfVectorizer()
        self.idea_categories = set()

    def generate_business_ideas(self, num_ideas: int = 10) -> List[Dict[str, str]]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_idea = {executor.submit(self._generate_single_idea): i for i in range(num_ideas)}
            ideas = []
            for future in concurrent.futures.as_completed(future_to_idea):
                idea = future.result()
                if idea:
                    ideas.append(idea)

        unique_ideas = self._filter_unique_ideas(ideas)
        categorized_ideas = self._categorize_ideas(unique_ideas)

        print(f"生成的创意数量: {len(unique_ideas)}")
        print(f"分类后的创意: {[(k, len(v)) for k, v in categorized_ideas.items()]}")

        selected_ideas = self._select_diverse_ideas(categorized_ideas)
        return selected_ideas

    def _generate_single_idea(self) -> Dict[str, str]:
        dialogue = self._generate_dialogue()
        summary = self._summarize_dialogue(dialogue)
        business_opportunity = self._extract_business_opportunity(summary)
        return business_opportunity

    import random

    def _generate_dialogue(self) -> str:
        roles = {
            "创业者": """你是一位充满激情的连续创业者，有多次成功和失败的经验。你具有敏锐的洞察力，强烈的问题解决导向，有说服力的沟通能力，高度的适应性和抗压能力，以及扎实的商业知识。在讨论中，你应该展现出对机会的热情，同时也要表现出对风险的认知。""",
            "投资人": """你是一位在科技和创新领域有10年以上经验的风险投资人。你有敏锐的商业嗅觉，深厚的行业知识，严谨的分析能力，广泛的人脉网络，和战略性思维。在对话中，你应该提出尖锐的问题，挑战假设，并要求具体的数据支持。""",
            "技术专家": """你是一位在多个前沿技术领域都有深入研究的资深工程师，拥有20年以上的经验。你有深厚的技术功底，强大的问题解决能力，前瞻性思维，实用主义态度，和跨学科知识。在讨论中，你应该提供深入的技术见解，评估技术可行性，并指出潜在的技术挑战。""",
            "市场营销专家": """你是一位在多个行业有丰富经验的高级市场营销顾问。你有出色的市场洞察力，创新的营销策略思维，数据驱动的决策能力，多渠道整合能力，和敏锐的竞争分析能力。在对话中，你应该提供市场趋势分析，讨论目标受众定位，并提出具体的营销策略建议。""",
            "潜在用户": """你是一位典型的目标市场代表，对新产品和服务有着浓厚的兴趣，但也有着明确的需求和担忧。你是一个理性消费者，注重用户体验，关注隐私和安全，追求个性化，并且是社交媒体活跃用户。在讨论中，你应该表达出对产品或服务的实际需求，提出潜在的使用场景，并指出可能的顾虑或改进建议。"""
        }

        dialogue = ""
        context = "一个创新的智能家居系统，集成了AI助手、能源管理和安全监控功能。"

        for round in range(5):
            round_dialogue = ""
            # 随机化角色顺序
            shuffled_roles = list(roles.items())
            random.shuffle(shuffled_roles)

            for role, description in shuffled_roles:
                prompt = f"""
                你现在扮演的角色是：{role}

                角色描述：
                {description}

                当前讨论的主题是：{context}
                这是第{round + 1}轮对话。请根据你的角色特点，对当前讨论的主题发表看法或提出问题。
                你的回应应该围绕以下几个方面：问题识别、解决方案讨论、市场分析、技术可行性、用户需求等。

                之前的对话记录：
                {dialogue}

                请用约150-200字的篇幅给出你的回应。直接输出对话内容，不需要额外的解释或格式。以你的角色名开头。
                确保你的观点和建议都与你的背景和专业知识一致，并对之前的讨论有所回应或拓展。
                """
                result = self.llm_processor.process_llm_request(prompt, "dialogue", self.llm_processor.model)
                round_dialogue += result + "\n\n"

            dialogue += round_dialogue

            # 更新上下文，为下一轮对话做准备
            context_prompt = f"""
            基于以下对话内容，总结讨论的主要点和新出现的想法，用一到两句话描述下一轮讨论应该关注的方向：

            {round_dialogue}
            """
            context = self.llm_processor.process_llm_request(context_prompt, "summary", self.llm_processor.model)

        return dialogue

    def _summarize_dialogue(self, dialogue: str) -> str:
        prompt = f"""
        请总结以下多轮对话中讨论的主要商业想法：

        {dialogue}

        总结应包括：
        1. 识别的问题或需求
        2. 提出的解决方案
        3. 潜在的市场机会
        4. 主要的技术或实施挑战
        5. 关键的讨论点或分歧

        请提供一个简洁但全面的总结。
        """
        result = self.llm_processor.process_llm_request(prompt, "summary", self.llm_processor.model)
        return result

    def _extract_business_opportunity(self, summary: str) -> Dict[str, str]:
        prompt = f"""
        基于以下对话总结，提取一个具体的商业机会：

        {summary}

        请以JSON格式输出，包含以下字段：
        {{
            "problem": "识别的核心问题或需求",
            "solution": "提出的创新解决方案",
            "business_plan": "初步的商业计划概述",
            "category": "该创意所属的广泛类别或行业"
        }}
        """
        result = self.llm_processor.process_llm_request(prompt, "extract", self.llm_processor.model)
        try:
            business_opportunity = json.loads(_custom_parser(result))
            return business_opportunity
        except Exception as e:
            print(f"提取商业机会时发生错误: {e}")
            return {"problem": "", "solution": "", "business_plan": "", "category": ""}

    # def generate_business_ideas(self, num_ideas: int = 10) -> List[Dict[str, str]]:
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    #         future_to_idea = {executor.submit(self._generate_single_idea): i for i in range(num_ideas)}
    #         ideas = []
    #         for future in concurrent.futures.as_completed(future_to_idea):
    #             idea = future.result()
    #             if idea:
    #                 ideas.append(idea)
    #
    #     unique_ideas = self._filter_unique_ideas(ideas)
    #     categorized_ideas = self._categorize_ideas(unique_ideas)
    #
    #     print(f"生成的创意数量: {len(unique_ideas)}")
    #     print(f"分类后的创意: {[(k, len(v)) for k, v in categorized_ideas.items()]}")
    #
    #     selected_ideas = self._select_diverse_ideas(categorized_ideas)
    #     return selected_ideas
    #
    # def _generate_single_idea(self) -> Dict[str, str]:
    #     prompt = self._get_generate_prompt()
    #     try:
    #         result = self.llm_processor.process_llm_request(prompt, "create", self.llm_processor.model)
    #         problem = result.strip()
    #         return self.generate_solution_for_problem(problem)
    #     except Exception as e:
    #         print(f"生成创意时发生错误: {e}")
    #         return None

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

    def _categorize_ideas(self, ideas: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        texts = [f"{idea['problem']} {idea['solution']} {idea['business_plan']}" for idea in ideas]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        num_clusters = min(len(ideas), 5)  # 最多5个类别
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        categorized_ideas = {}
        for i, label in enumerate(cluster_labels):
            category = f"Category_{label}"
            if category not in categorized_ideas:
                categorized_ideas[category] = []
            categorized_ideas[category].append(ideas[i])

        return categorized_ideas

    def _select_diverse_ideas(self, categorized_ideas: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        selected_ideas = []
        all_evaluated_ideas = []
        for category, ideas in categorized_ideas.items():
            evaluated_ideas = self.evaluate_ideas(ideas)
            all_evaluated_ideas.extend(evaluated_ideas)
            if evaluated_ideas:
                try:
                    best_idea = max(evaluated_ideas, key=lambda x: x[2])
                    selected_ideas.append(best_idea[0])
                except ValueError as e:
                    print(f"错误发生在类别 {category}")
                    print(f"评估的创意: {evaluated_ideas}")
                    raise e

        self._save_evaluation_results(all_evaluated_ideas)
        return selected_ideas
    def generate_solution_for_problem(self, problem: str) -> Dict[str, str]:
        prompt = self._get_solution_prompt(problem)
        try:
            result = self.llm_processor.process_llm_request(prompt, "solve", self.llm_processor.model)
            evaluation_result = _custom_parser(result)
            solution_data = json.loads(evaluation_result)
            return {
                "problem": problem,
                "solution": solution_data["解决方案"],
                "business_plan": solution_data["商业方案"],
                "category": solution_data["类别"]
            }
        except Exception as e:
            print(f"为问题 '{problem}' 生成解决方案时发生错误: {e}")
            return {"problem": problem, "solution": "", "business_plan": "", "category": ""}

    def evaluate_ideas(self, ideas: List[Dict[str, str]]) -> List[Tuple[Dict[str, str], str, float]]:
        evaluated_ideas = []
        for idea in ideas:
            prompt = self._get_evaluate_prompt(idea)
            try:
                result = self.llm_processor.process_llm_request(prompt, "evaluate", self.llm_processor.model)
                evaluation_result = _custom_parser(result)
                data = json.loads(evaluation_result)
                if isinstance(data, dict) and '总分' in data:
                    score = float(data.get('总分', 0))  # 确保分数是浮点数
                    diversity_score = self._calculate_diversity_score(idea)
                    adjusted_score = float((score + diversity_score) / 2)  # 确保结果是浮点数
                    evaluated_ideas.append(
                        (idea, json.dumps(evaluation_result, ensure_ascii=False, indent=4), adjusted_score))
                    print(f"评估结果: 问题 = {idea['problem'][:30]}..., 分数 = {adjusted_score}")
            except Exception as e:
                print(f"评估创意 '{idea['problem']}' 时发生错误: {e}")
        return evaluated_ideas

    def _calculate_diversity_score(self, new_idea: Dict[str, str]) -> float:
        if not self.generated_ideas:
            return 10.0  # 如果是第一个创意，给予最高的多样性分数

        new_text = f"{new_idea['problem']} {new_idea['solution']} {new_idea['business_plan']}"
        existing_texts = [f"{idea['problem']} {idea['solution']} {idea['business_plan']}" for idea in
                          self.generated_ideas]

        all_texts = existing_texts + [new_text]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

        avg_similarity = float(np.mean(cosine_similarities))  # 确保是浮点数
        diversity_score = float((1 - avg_similarity) * 10)  # 确保结果是浮点数
        return diversity_score

    def _get_generate_prompt(self) -> str:
        excluded_categories = ", ".join(self.idea_categories)
        return f"""
        请生成一个创新的商业创意或问题陈述。这个创意应该满足以下条件：
        1. 解决真实世界的问题或满足特定需求。
        2. 具有独特性和创新性。
        3. 不属于以下已经探索过的类别：{excluded_categories}
        4. 尽可能涉及新的行业或领域。

        请直接输出问题陈述，不需要额外的解释或格式。
        """

    def _get_solution_prompt(self, problem: str) -> str:
        return f"""
        针对以下问题，请提供一个创新的解决方案和简要的商业方案：

        问题：{problem}

        请以JSON格式输出，包含以下字段：
        {{
            "解决方案": "详细描述你的创新解决方案",
            "商业方案": "简要说明如何将这个解决方案商业化",
            "类别": "为这个创意分配一个广泛的类别或行业"
        }}
        """

    def _get_evaluate_prompt(self, idea: Dict[str, str]) -> str:
        return f"""
        任务：全面评估以下商业创意的潜力和可行性。

        问题：{idea['problem']}
        解决方案：{idea['solution']}
        商业方案：{idea['business_plan']}
        类别：{idea['category']}

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

    def _save_evaluation_results(self, evaluated_ideas: List[Tuple[Dict[str, str], str, float]]):
        sorted_ideas = sorted(evaluated_ideas, key=lambda x: x[2], reverse=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"business_ideas_evaluation_{timestamp}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            for idea, evaluation, score in sorted_ideas:
                f.write(f"问题: {idea['problem']}\n")
                f.write(f"解决方案: {idea['solution']}\n")
                f.write(f"商业方案: {idea['business_plan']}\n")
                f.write(f"类别: {idea['category']}\n")
                f.write(f"评分: {score}\n")
                f.write(f"评估: {evaluation}\n")
                f.write("-" * 50 + "\n")

        print(f"评估结果已保存到文件: {filename}")

def main():
    db_path = "llm_results.db"
    openai_base_url = "http://localhost:5000/v1"
    openai_api_key = "810001a0a02948d5bf640a98cb69f653"

    llm_processor = LLMProcessor(db_path, openai_base_url, openai_api_key)
    idea_processor = BusinessIdeaProcessor(llm_processor)

    ideas = idea_processor.generate_business_ideas(num_ideas=10)
    evaluated_ideas = idea_processor.evaluate_ideas(ideas)

    # 按评分排序并打印结果
    sorted_ideas = sorted(evaluated_ideas, key=lambda x: x[2], reverse=True)
    for idea, evaluation, score in sorted_ideas:
        print(f"问题: {idea['problem']}")
        print(f"解决方案: {idea['solution']}")
        print(f"商业方案: {idea['business_plan']}")
        print(f"类别: {idea['category']}")
        print(f"评分: {score}")
        print(f"评估: {evaluation}")
        print("-" * 50)

if __name__ == "__main__":
    main()