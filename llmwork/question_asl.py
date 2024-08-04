import queue
import random
import re
import threading

from llmwork.llm_job import LLMProcessor


class QASystem:
    def __init__(self, llm_processor):
        self.llm_processor = llm_processor
        self.model = self.llm_processor.get_available_models()
        self.question_queue = queue.Queue()
        self.answer_queue = queue.Queue()
        self.critique_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.generated_methods = set()  # 使用集合来存储唯一的方案

    def extract_core_methods(self, summary):
        # 使用正则表达式提取核心方案
        match = re.search(r'1\.\s*核心方案：(.+?)(?=\n2\.|\Z)', summary, re.DOTALL)
        if match:
            core_methods = match.group(1).strip()
            # 分割多个方案（假设方案之间用分号或逗号分隔）
            methods = re.split(r'[;,；，]', core_methods)
            # 清理和规范化每个方案
            methods = [method.strip() for method in methods if method.strip()]
            return methods
        return []

    def generate_summary(self, question, improved_answer, question_id):
        prompt = f"""基于以下问题和改进后的答案，请提供一个简洁的总结，重点关注可行的商业方案：

    思考方向: {question}
    改进后的商业模式: {improved_answer}

    请按以下格式输出：
    1. 核心方案：(提炼1-2个最具可行性的商业方案，每个50字以内)
    2. 实施建议：(针对核心方案给出2-3点具体的实施建议，每点30字以内)
    3. 潜在影响：(分析这些方案可能带来的市场影响，50字以内)"""

        summary = self.llm_processor.process_llm_request(prompt, question_id, self.model)
        # 提取核心方案并更新生成的方法集合
        new_methods = self.extract_core_methods(summary)
        self.generated_methods.update(new_methods)
        return summary.strip()

    def generate_question(self, topic, question_id):
        # 获取已生成的方案列表
        existing_methods = list(self.generated_methods)

        # 随机选择1-2个已有方案（如果有的话）
        selected_methods = random.sample(existing_methods, min(100, len(existing_methods)))

        methods_prompt = ""
        if selected_methods:
            methods_prompt = f"考虑以下已有的商业方案，请提出新的方案：{', '.join(selected_methods)}。"

        prompts = [
            f"""作为一位创新思维专家，请基于主题'{topic}'提出一个富有洞察力和启发性的问题。{methods_prompt}你的问题应该：
            1. 挑战当前行业的核心假设，揭示潜在的盲点或机会
            2. 探索看似不可能但潜力巨大的商业概念
            3. 鼓励从全新角度思考问题，打破常规思维模式
            4. 启发人们考虑小众市场或未被满足需求的巨大潜力
            5. 结合当前技术趋势和社会变化，预见未来可能出现的商业机会
            6. 考虑跨界创新的可能性，融合不同行业的优势
            请直接给出思考方向，确保问题简洁明了，富有挑战性和前瞻性。不需要其他解释。"""
        ]

        # 随机选择一个提示
        prompt = random.choice(prompts)

        question = self.llm_processor.process_llm_request(prompt, question_id, self.model)
        return question.strip()

    def generate_answer(self, question, question_id, topic):
        # 在生成答案时考虑已有的方案
        existing_methods = ', '.join(self.generated_methods)
        prompt = f"""请针对以下问题提供一个全面、具有洞察力的答案，聚焦于主题'{topic}'。你的回答应该：
        1. 提供3-4个具体可行的商业方案，确保与以下已生成的方案不重复：
           {existing_methods}
        2. 每个方案必须包括：
           a) 简要描述（不超过50字）
           b) 目标市场和客户群体
           c) 独特的价值主张
           d) 潜在的商业模式
           e) 可能面临的挑战及初步应对策略
        3. 确保方案具有创新性、可扩展性和实际可行性
        4. 考虑当前市场趋势、技术发展和社会变化
        5. 尽可能提供数据支持或案例参考
        6. 考虑方案的长期可持续性和社会影响

        思考方向：{question}

        请以结构化的方式呈现你的答案，使用标题和子标题来组织内容。确保每个方案都是独特且有价值的。"""
        answer = self.llm_processor.process_llm_request(prompt, question_id, self.model)
        return answer.strip()

    def critique_answer(self, question, answer, question_id, topic):
        prompt = f"""请对以下问题和答案进行严格、深入的批评和分析：

        思考方向：{question}
        可能的商业模式：{answer}
        主题：{topic}

        请从以下角度进行尖锐而务实的批评：
        1. 与主题的相关性和切入点的准确性
        2. 商业模式的创新性和独特价值主张
        3. 市场需求的真实性和规模潜力
        4. 财务可行性和长期盈利能力
        5. 实施难度和所需资源
        6. 竞争优势和进入壁垒
        7. 技术可行性和依赖度
        8. 法律合规性和潜在的监管风险
        9. 社会影响和可持续发展
        10. 可扩展性和全球化潜力
        11. 与现有生态系统的整合度
        12. 对客户痛点的解决程度

        请列出所有需要改进的要点，格式如下：
        1. [改进项]: 简明扼要地指出问题所在，不超过30个字。
        2. [改进项]: 简明扼要地指出问题所在，不超过30个字。
        ...

        注意：
        - 保持批评的客观性和建设性，但要直接指出问题
        - 重点关注方案的实际可行性和市场潜力
        - 特别注意方案是否真正解决了问题，而不是在重复已有的解决方案
        - 评估方案是否充分考虑了现实世界的复杂性和潜在障碍
        - 考虑方案在不同市场环境和地理位置的适用性
        - 评估方案是否充分利用了新兴技术或市场趋势
        - 如果方案忽视了重要的利益相关者或潜在的负面影响，请指出
        - 在商业和财务可行性方面要特别严格，任何不切实际的假设都应被质疑

        如果真的找不到任何值得批评的地方，可以说"该答案整体表现良好，但仍有提升空间"，然后列出一两个可以进一步完善的小细节。"""

        critique = self.llm_processor.process_llm_request(prompt, question_id, self.model)
        return critique.strip()

    def improve_answer(self, question, original_answer, critique, question_id, topic):
        prompt = f"""请根据以下信息改进答案：

问题：{question}
原始答案：{original_answer}
改进要点：{critique}
主题：{topic}

请提供一个改进后的、更加全面和准确的答案。注意处理所有列出的改进要点，但不要逐点回应批评意见。"""

        improved_answer = self.llm_processor.process_llm_request(prompt, question_id, self.model)
        return improved_answer.strip()

    def question_generator(self, topic, num_rounds):
        for i in range(num_rounds):
            if self.stop_event.is_set():
                break
            question_id = f"q_{i + 1}"
            question = self.generate_question(topic, question_id)
            self.question_queue.put((question, question_id, topic))
        self.question_queue.put(None)  # 表示问题生成结束

    def answer_generator(self):
        while not self.stop_event.is_set():
            item = self.question_queue.get()
            if item is None:
                break
            question, question_id, topic = item
            answer = self.generate_answer(question, question_id, topic)
            self.answer_queue.put((question, answer, question_id, topic))
        self.answer_queue.put(None)  # 表示回答生成结束

    def critique_and_improve(self):
        while not self.stop_event.is_set():
            item = self.answer_queue.get()
            if item is None:
                break
            question, answer, question_id, topic = item
            critique = self.critique_answer(question, answer, question_id, topic)
            improved_answer = self.improve_answer(question, answer, critique, question_id, topic)
            summary = self.generate_summary(question, improved_answer, question_id)
            self.critique_queue.put((question, answer, critique, improved_answer, summary, question_id))
        self.critique_queue.put(None)  # 表示批评和改进结束

    def self_qa(self, topic, num_rounds=3):
        question_thread = threading.Thread(target=self.question_generator, args=(topic, num_rounds))
        answer_thread = threading.Thread(target=self.answer_generator)
        critique_thread = threading.Thread(target=self.critique_and_improve)

        question_thread.start()
        answer_thread.start()
        critique_thread.start()

        round_num = 1
        while True:
            item = self.critique_queue.get()
            if item is None:
                break
            question, original_answer, critique, improved_answer, summary, question_id = item
            print(f"\n--- 第 {round_num} 轮问答 (ID: {question_id}) ---")
            print(f"问题: {question}")
            print(f"原始答案: {original_answer}")
            print(f"改进要点: {critique}")
            print(f"改进后的答案: {improved_answer}")
            print(f"\n总结与建议:\n{summary}")

            # 打印当前所有生成的方法
            print("\n当前生成的所有独特方案:")
            for i, method in enumerate(self.generated_methods, 1):
                print(f"{i}. {method}")

            round_num += 1

        question_thread.join()
        answer_thread.join()
        critique_thread.join()

    def stop(self):
        self.stop_event.set()


# 使用示例
if __name__ == "__main__":
    db_path = "llm_results.db"
    openai_base_url = "http://localhost:5000/v1"
    openai_api_key = "810001a0a02948d5bf640a98cb69f653"

    llm_processor = LLMProcessor(db_path, openai_base_url, openai_api_key)
    qa_system = QASystem(llm_processor)

    topic = "银发经济"

    try:
        qa_system.self_qa(topic, num_rounds=1000)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        qa_system.stop()
        llm_processor.wait_for_completion()
        llm_processor.stop()
