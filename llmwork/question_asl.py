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

    问题: {question}
    改进后的答案: {improved_answer}

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
        selected_methods = random.sample(existing_methods, min(10, len(existing_methods)))

        methods_prompt = ""
        if selected_methods:
            methods_prompt = f"考虑以下已有的商业方案：{', '.join(selected_methods)}。"

        prompts = [
            f"""基于主题'{topic}'，{methods_prompt}请创造一个富有洞察力的问题。你的问题应该：
            1. 探索跨界融合的可能性，结合不同行业或技术
            2. 挑战传统思维，提出颠覆性的商业模式
            3. 考虑未来5-10年可能出现的新兴市场需求
            4. 聚焦于解决全球性挑战（如气候变化、人口老龄化等）
            请直接给出问题，无需其他解释。""",

            f"""想象你是一位来自2050年的时间旅行者。基于主题'{topic}'和你对未来的了解，{methods_prompt}提出一个当前人们可能忽视但在30年后将变得至关重要的问题。这个问题应该：
            1. 揭示一个潜在的巨大市场机会
            2. 涉及技术、社会和环境的交叉领域
            3. 激发人们对长期可持续发展的思考
            请直接给出问题，无需其他解释。""",

            f"""作为一个创新思维专家，请基于主题'{topic}'提出一个反直觉的问题。{methods_prompt}这个问题应该：
            1. 挑战当前行业的核心假设
            2. 探索看似不可能的商业概念
            3. 鼓励从全新角度思考问题
            4. 启发人们考虑小众市场的巨大潜力
            请直接给出问题，无需其他解释。""",

            f"""想象一个由AI、量子计算和生物技术主导的世界。基于主题'{topic}'，{methods_prompt}提出一个探索这种未来情境下商业机会的问题。这个问题应该：
            1. 考虑技术融合带来的革命性变化
            2. 探讨人类需求和价值观的潜在转变
            3. 涉及新兴的伦理和社会挑战
            4. 激发对全新商业生态系统的想象
            请直接给出问题，无需其他解释。""",

            f"""基于主题'{topic}'，{methods_prompt}设想一个'黑天鹅'事件将在未来5年内发生。提出一个问题，探讨：
            1. 这个意外事件可能如何彻底改变当前的商业格局
            2. 哪些新的商业机会可能因此产生
            3. 企业如何在这种剧变中生存并茁壮成长
            4. 如何将潜在的危机转化为创新的契机
            请直接给出问题，无需其他解释。""",

            f"""想象你是一位生活在2100年的企业家。回顾过去的'{topic}'领域，{methods_prompt}提出一个关于以下方面的问题：
            1. 哪些当前被忽视的小趋势最终成为了改变游戏规则的大趋势
            2. 在这个未来世界，人类的核心需求和价值观发生了哪些根本性的转变
            3. 哪些现在看似不可能的商业模式在未来变得司空见惯
            4. 技术、社会和环境的协同进化如何创造出全新的商业生态系统
            请直接给出问题，无需其他解释。""",

            f"""作为一个跨界创新专家，{methods_prompt}请基于主题'{topic}'提出一个将看似不相关的行业或概念结合起来的问题。这个问题应该：
            1. 探索两个或更多截然不同领域的意外交集
            2. 挑战传统的行业界限和分类
            3. 揭示隐藏的协同效应和创新机会
            4. 启发人们思考如何将不同专业知识和资源整合以创造新价值
            请直接给出问题，无需其他解释。""",

            f"""设想一个后稀缺社会，其中能源、食物和基本资源都变得极其丰富。基于主题'{topic}'，{methods_prompt}提出一个问题，探讨：
            1. 在这样的世界里，商业和经济的根本动力将如何改变
            2. 哪些全新的人类需求和欲望可能会出现
            3. 企业如何在一个不再由稀缺驱动的市场中创造价值和利润
            4. 这种转变可能带来的社会、伦理和哲学挑战
            请直接给出问题，无需其他解释。""",

            f"""想象一个AI已经达到超人类水平的世界。关于主题'{topic}'，{methods_prompt}提出一个问题，考虑：
            1. 人类在这样的商业环境中的独特价值和不可替代性
            2. 可能出现的全新商业模式和价值创造方式
            3. 人机协作如何重塑企业组织结构和决策过程
            4. 在这种背景下如何重新定义创新、创造力和企业家精神
            请直接给出问题，无需其他解释。""",

            f"""作为一个系统思考专家，请基于主题'{topic}'提出一个关于循环经济的问题。{methods_prompt}这个问题应该：
            1. 探讨如何将线性商业模式转变为闭环系统
            2. 考虑跨行业和跨供应链的资源循环利用机会
            3. 挑战传统的"生产-消费-废弃"模式
            4. 思考如何在追求可持续性的同时保持经济增长和创新
            请直接给出问题，无需其他解释。""",

            f"""想象一个去中心化自治组织(DAO)主导的商业世界。关于主题'{topic}'，{methods_prompt}提出一个问题，探讨：
            1. 传统企业结构和管理模式可能面临的颠覆
            2. 新型协作和决策机制带来的机遇和挑战
            3. 区块链和智能合约如何重塑商业关系和信任机制
            4. 在这种新范式下，价值创造和分配方式的根本性变革
            请直接给出问题，无需其他解释。""",

            f"""设想一个基因编辑技术被广泛应用的未来。基于主题'{topic}'，{methods_prompt}提出一个问题，考虑：
            1. 这项技术可能如何彻底改变特定行业或创造全新产业
            2. 由此产生的前所未有的伦理和监管挑战
            3. 个性化和定制化在这种背景下的新含义
            4. 生物技术与其他新兴技术（如AI、纳米技术）的融合可能带来的革命性变革
            请直接给出问题，无需其他解释。""",

            f"""作为一个未来学家，请基于主题'{topic}'提出一个关于应对气候变化的创新商业模式的问题。{methods_prompt}这个问题应该：
            1. 探讨如何将环境可持续性转化为强大的竞争优势
            2. 考虑跨行业合作应对全球性挑战的可能性
            3. 思考如何平衡短期商业利益和长期生态影响
            4. 探索气候适应性和减缓策略如何催生新的商业机会
            请直接给出问题，无需其他解释。""",

            f"""想象一个虚拟现实和增强现实技术高度发达的世界。关于主题'{topic}'，{methods_prompt}提出一个问题，探讨：
            1. 物理和数字世界的界限模糊可能带来的新商业模式
            2. 沉浸式体验如何重新定义产品、服务和客户互动
            3. 虚拟经济与实体经济的交织可能产生的新机遇和挑战
            4. 这种技术如何改变工作、学习和社交的本质，以及由此产生的商业implications
            请直接给出问题，无需其他解释。""",
            f"""考虑到数字鸿沟和技术普及的不平等，基于主题'{topic}'，{methods_prompt}提出一个问题，探讨：
            1. 如何为技术落后地区或群体开发适合的商业模式
            2. 在保证商业可持续性的同时，如何促进技术普惠
            3. 针对低收入群体的创新产品或服务可能带来的社会影响
            4. 如何平衡商业利益和缩小数字鸿沟的社会责任
            请直接给出问题，无需其他解释。""",

            f"""想象一个社会分层日益严重的未来。关于主题'{topic}'，{methods_prompt}提出一个问题，考虑：
            1. 针对不同社会阶层的差异化商业策略
            2. 如何在追求利润的同时促进社会流动性
            3. 面向底层群体的创新商业模式可能带来的机遇和挑战
            4. 企业在缓解社会不平等方面可能扮演的角色
            请直接给出问题，无需其他解释。""",

            f"""考虑到全球范围内的贫富差距，基于主题'{topic}'，{methods_prompt}提出一个问题，探讨：
            1. 如何为经济能力不充足的群体设计可负担的产品或服务
            2. 在新兴市场和发展中国家可能存在的独特商业机会
            3. 如何通过商业创新改善第三世界国家的生活质量
            4. 跨国公司在促进全球经济平等中可能扮演的角色
            请直接给出问题，无需其他解释。""",

            f"""想象一个贫富差距持续扩大的未来。关于主题'{topic}'，{methods_prompt}提出一个问题，探讨：
            1. 如何开发既能服务富裕群体又不忽视中低收入群体的商业模式
            2. 在追求利润的同时，如何通过商业创新促进经济机会的公平分配
            3. 针对不同收入群体的差异化定价策略可能带来的影响和挑战
            4. 如何利用新技术或创新商业模式来缩小贫富差距
            请直接给出问题，无需其他解释。""",

            f"""考虑到气候变化对不同社会群体的不均衡影响，基于主题'{topic}'，{methods_prompt}提出一个问题，探讨：
            1. 如何开发既环保又经济实惠的产品或服务，以服务低收入群体
            2. 在应对气候变化的同时，如何创造面向弱势群体的就业机会
            3. 如何设计商业模式来帮助脆弱社区提高气候适应能力
            4. 企业如何在追求可持续发展的过程中考虑社会公平问题
            请直接给出问题，无需其他解释。""",

            f"""作为一位跨时代的创新思想家，请基于主题'{topic}'构建一个多维度的问题。{methods_prompt}你的问题应该：
        1. 融合至少两个看似不相关的领域（例如生物技术和城市规划）
        2. 考虑短期（1-3年）、中期（5-10年）和长期（20+年）的影响
        3. 探讨技术进步可能带来的意外社会后果
        4. 挑战一个当前行业的核心假设
        5. 考虑全球性挑战（如气候变化、不平等）对该主题的影响
        请直接给出问题，无需其他解释。""",

            f"""想象你是来自2100年的时间旅行者，回顾'{topic}'领域的发展。{methods_prompt}请提出一个问题，该问题应：
        1. 揭示一个当前被忽视但最终改变游戏规则的小趋势
        2. 探讨如何在保持经济增长的同时实现真正的可持续发展
        3. 考虑技术、社会规范和人类价值观的协同演变
        4. 反思哪些现在看似革命性的创新最终成为了昙花一现
        5. 分析在应对全球性危机中，商业创新扮演了什么角色
        请直接给出问题，无需其他解释。""",

            f"""作为一个系统思考专家，请基于主题'{topic}'提出一个探索悖论或矛盾的问题。{methods_prompt}这个问题应该：
        1. 考虑看似相互冲突的目标（如盈利增长和环境保护）如何能够协同实现
        2. 探讨在推动创新的同时如何管理相关的伦理风险
        3. 分析技术进步可能如何同时解决和加剧社会问题
        4. 思考在日益分化的世界中如何构建包容性的商业模式
        5. 探索如何在保持企业竞争力的同时促进行业内的开放协作
        请直接给出问题，无需其他解释。""",

            f"""设想一个AI、基因编辑和量子计算技术高度发达的世界。关于主题'{topic}'，{methods_prompt}提出一个问题，探讨：
        1. 这些技术的融合如何彻底重塑商业和社会结构
        2. 在这样的世界里，人类独特价值和创造力的体现
        3. 如何重新定义工作、教育和个人发展的概念
        4. 潜在的新伦理挑战和所需的新型监管框架
        5. 如何确保技术进步的成果能够公平地惠及所有社会群体
        请直接给出问题，无需其他解释。""",

            f"""想象一个后稀缺社会，但面临严重的人口老龄化和气候危机。基于主题'{topic}'，{methods_prompt}提出一个问题，考虑：
        1. 在资源丰富但环境受限的情况下，商业模式将如何演变
        2. 如何重新定义价值创造，使其超越传统的经济指标
        3. 老年人口可能成为创新和价值创造的新源泉
        4. 如何设计跨代际合作的商业生态系统
        5. 在这种情境下，企业的社会责任和盈利目标如何重新平衡
        请直接给出问题，无需其他解释。"""
        ]

        # 随机选择一个提示
        prompt = random.choice(prompts)

        question = self.llm_processor.process_llm_request(prompt, question_id, self.model)
        return question.strip()

    def generate_answer(self, question, question_id):
        # 在生成答案时考虑已有的方案
        existing_methods = ', '.join(self.generated_methods)
        prompt = f"""请针对以下问题提供一个全面、具有洞察力的答案。你的回答应该：
        1. 提供2-3个具体可行的商业方案，避免与以下已生成的方案重复：
           {existing_methods}
        2. 每个方案包括：
           a) 简要描述
           b) 目标市场
           c) 潜在优势
           d) 可能面临的挑战
        3. 考虑创新性和可扩展性
        4. 结合实际市场情况和趋势

        问题：{question}"""
        answer = self.llm_processor.process_llm_request(prompt, question_id, self.model)
        return answer.strip()

    def critique_answer(self, question, answer, question_id):
        prompt = f"""请对以下问题和答案进行严格、深入的批评和分析：

    问题：{question}
    答案：{answer}

    请从以下角度进行尖锐而务实的批评：
    1. 商业模式的可行性和可持续性
    2. 市场潜力和竞争优势
    3. 财务可行性和盈利模式
    4. 创新程度和独特性
    5. 实施难度和潜在风险
    6. 法律和道德合规性
    7. 社会影响和责任
    8. 技术可行性和成熟度
    9. 对现有商业生态系统的影响
    10. 宏观经济和行业趋势的契合度

    请列出所有需要改进的要点，格式如下：
    1. [改进项]: 简明扼要地指出问题所在，不超过25个字。
    2. [改进项]: 简明扼要地指出问题所在，不超过25个字。
    ...

    注意：
    - 保持批评的客观性和建设性，但不要客气或婉转。
    - 只指出问题，不要提供解决方案。
    - 如果发现明显的逻辑漏洞、假设错误或不切实际的想法，请直接指出。
    - 特别关注该想法是否真正解决了问题，而不是在重复已有的解决方案。
    - 评估该想法是否充分考虑了现实世界的复杂性和潜在障碍。
    - 如果该想法缺乏创新性或只是对现有模式的简单模仿，请明确指出。
    - 考虑该想法在不同市场环境和地理位置的适用性。
    - 评估该想法是否充分利用了新兴技术或市场趋势。
    - 如果该想法忽视了重要的利益相关者或潜在的负面影响，请指出。
    - 在商业和财务可行性方面要特别严格，任何不切实际的假设都应被质疑。

    如果真的找不到任何值得批评的地方，可以说"该答案整体表现良好，但仍有提升空间"，然后列出一两个可以进一步完善的小细节。"""

        critique = self.llm_processor.process_llm_request(prompt, question_id, self.model)
        return critique.strip()

    def improve_answer(self, question, original_answer, critique, question_id):
        prompt = f"""请根据以下信息改进答案：

问题：{question}
原始答案：{original_answer}
改进要点：{critique}

请提供一个改进后的、更加全面和准确的答案。注意处理所有列出的改进要点，但不要逐点回应批评意见。"""

        improved_answer = self.llm_processor.process_llm_request(prompt, question_id, self.model)
        return improved_answer.strip()

    def question_generator(self, topic, num_rounds):
        for i in range(num_rounds):
            if self.stop_event.is_set():
                break
            question_id = f"q_{i + 1}"
            question = self.generate_question(topic, question_id)
            self.question_queue.put((question, question_id))
        self.question_queue.put(None)  # 表示问题生成结束

    def answer_generator(self):
        while not self.stop_event.is_set():
            item = self.question_queue.get()
            if item is None:
                break
            question, question_id = item
            answer = self.generate_answer(question, question_id)
            self.answer_queue.put((question, answer, question_id))
        self.answer_queue.put(None)  # 表示回答生成结束

    def critique_and_improve(self):
        while not self.stop_event.is_set():
            item = self.answer_queue.get()
            if item is None:
                break
            question, answer, question_id = item
            critique = self.critique_answer(question, answer, question_id)
            improved_answer = self.improve_answer(question, answer, critique, question_id)
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


# 使用示例部分保持不变...

# 使用示例
if __name__ == "__main__":
    db_path = "llm_results.db"
    openai_base_url = "http://localhost:5000/v1"
    openai_api_key = "810001a0a02948d5bf640a98cb69f653"

    llm_processor = LLMProcessor(db_path, openai_base_url, openai_api_key)
    qa_system = QASystem(llm_processor)

    topic = "日本的银发经济有哪些项目可以借鉴"

    try:
        qa_system.self_qa(topic, num_rounds=1000)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        qa_system.stop()
        llm_processor.wait_for_completion()
        llm_processor.stop()
