import queue
import threading
import time

from llmwork.llm_job import LLMProcessor


class QASystem:
    def __init__(self, llm_processor):
        self.llm_processor = llm_processor
        self.model = self.llm_processor.get_available_models()
        self.question_queue = queue.Queue()
        self.answer_queue = queue.Queue()
        self.critique_queue = queue.Queue()
        self.stop_event = threading.Event()

    def generate_question(self, topic):
        prompt = f"请基于主题'{topic}'创作一个有趣且富有洞察力的问题。"
        question = self.llm_processor.process_llm_request(prompt, self.model)
        return question.strip()

    def generate_answer(self, question):
        prompt = f"请回答以下问题，提供详细且有见地的答案：{question}"
        answer = self.llm_processor.process_llm_request(prompt, self.model)
        return answer.strip()

    def critique_answer(self, question, answer):
        prompt = f"""请对以下问题和答案进行严格的批评和分析：

问题：{question}
答案：{answer}

请仅列出需要改进的要点，格式如下：
1. [改进项]: 简要说明为什么需要改进，但不要给出具体的改动建议。
2. [改进项]: 简要说明为什么需要改进，但不要给出具体的改动建议。
...

注意：
- 只指出问题，不要提供解决方案。
- 保持简洁，每个改进项不超过20个字。
- 如果没有明显需要改进的地方，请说明"答案整体良好，无需大幅改动"。"""

        critique = self.llm_processor.process_llm_request(prompt, self.model)
        return critique.strip()

    def improve_answer(self, question, original_answer, critique):
        prompt = f"""请根据以下信息改进答案：

问题：{question}
原始答案：{original_answer}
改进要点：{critique}

请提供一个改进后的、更加全面和准确的答案。注意处理所有列出的改进要点，但不要逐点回应批评意见。"""

        improved_answer = self.llm_processor.process_llm_request(prompt, self.model)
        return improved_answer.strip()

    def question_generator(self, topic, num_rounds):
        for _ in range(num_rounds):
            if self.stop_event.is_set():
                break
            question = self.generate_question(topic)
            self.question_queue.put(question)
        self.question_queue.put(None)  # 表示问题生成结束

    def answer_generator(self):
        while not self.stop_event.is_set():
            question = self.question_queue.get()
            if question is None:
                break
            answer = self.generate_answer(question)
            self.answer_queue.put((question, answer))
        self.answer_queue.put(None)  # 表示回答生成结束

    def critique_and_improve(self):
        while not self.stop_event.is_set():
            item = self.answer_queue.get()
            if item is None:
                break
            question, answer = item
            critique = self.critique_answer(question, answer)
            improved_answer = self.improve_answer(question, answer, critique)
            self.critique_queue.put((question, answer, critique, improved_answer))
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
            question, original_answer, critique, improved_answer = item
            print(f"\n--- 第 {round_num} 轮问答 ---")
            print(f"问题: {question}")
            print(f"原始答案: {original_answer}")
            print(f"改进要点: {critique}")
            print(f"改进后的答案: {improved_answer}")
            round_num += 1

        question_thread.join()
        answer_thread.join()
        critique_thread.join()

    def stop(self):
        self.stop_event.set()


# 使用示例部分保持不变...

# 使用示例
if __name__ == "__main__":
    db_path = "llm_results.db"
    openai_base_url = "http://localhost:5000/v1"
    openai_api_key = "810001a0a02948d5bf640a98cb69f653"

    llm_processor = LLMProcessor(db_path, openai_base_url, openai_api_key)
    qa_system = QASystem(llm_processor)

    topic = "留在一线好还是二线城市好"

    try:
        qa_system.self_qa(topic, num_rounds=100)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        qa_system.stop()
        llm_processor.wait_for_completion()
        llm_processor.stop()