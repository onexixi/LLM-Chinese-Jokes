from typing import Dict, List, Any
import json

from llmwork.llm_job import LLMProcessor


class BusinessModelGenerator:
    def __init__(self, llm_processor):
        self.llm_processor = llm_processor
        self.business_model_canvas = [
            "customer_segments", "value_propositions", "channels",
            "customer_relationships", "revenue_streams", "key_resources",
            "key_activities", "key_partnerships", "cost_structure"
        ]

    def generate_business_model(self, industry: str, target_market: str) -> Dict[str, str]:
        prompts = self._create_prompts(industry, target_market)
        business_model = {}

        for key, prompt in prompts.items():
            response = self.llm_processor.process_llm_request(prompt)
            business_model[key] = response

        return business_model

    def _create_prompts(self, industry: str, target_market: str) -> Dict[str, str]:
        prompts = {
            "customer_segments": f"""
            针对{industry}行业的{target_market}市场，详细描述目标客户群:
            1. 人口统计特征 (年龄、性别、收入、教育程度等)
            2. 心理图谱 (生活方式、价值观、痛点、需求等)
            3. 行为特征 (购买习惯、决策过程、品牌偏好等)
            4. 细分市场 (列出2-3个具体的客户群体)
            对每个细分市场，提供详细描述和潜在规模估计。
            """,

            "value_propositions": f"""
            为{industry}行业的{target_market}市场创建强有力的价值主张:
            1. 核心问题: 客户面临的主要问题或未满足的需求是什么?
            2. 解决方案: 你的产品/服务如何解决这些问题?
            3. 独特卖点: 与竞争对手相比，你的解决方案有何独特之处?
            4. 主要好处: 客户使用你的产品/服务后会获得哪些具体好处?
            5. 痛点缓解: 你如何减轻客户的痛点或顾虑?
            请提供详细且引人注目的价值主张，突出你的竞争优势。
            """,

            "channels": f"""
            为{industry}行业的{target_market}市场设计有效的渠道策略:
            1. 客户接触点: 列出潜在客户可能接触到你的产品/服务的所有渠道。
            2. 销售渠道: 描述你将如何销售产品/服务（直销、代理商、在线等）。
            3. 交付渠道: 说明你将如何向客户交付产品/服务。
            4. 售后服务: 解释你将如何提供客户支持和售后服务。
            5. 渠道整合: 说明这些渠道如何相互配合，创造无缝的客户体验。
            对每个渠道，评估其效率、成本和对目标客户的覆盖率。
            """,

            "customer_relationships": f"""
            为{industry}行业的{target_market}市场制定客户关系策略:
            1. 客户获取: 描述如何吸引新客户。
            2. 客户保留: 解释如何维持现有客户关系。
            3. 客户培养: 说明如何增加客户的终身价值。
            4. 互动方式: 列出与客户互动的主要方式（个人协助、自助服务、社区等）。
            5. 客户体验: 描述理想的客户体验，以及如何在所有接触点实现它。
            6. 个性化: 解释如何为不同客户群提供个性化服务。
            提供具体的策略和工具，以建立长期、有利可图的客户关系。
            """,

            "revenue_streams": f"""
            为{industry}行业的{target_market}市场设计多元化的收入流:
            1. 主要收入来源: 列出并详细描述主要的收入来源。
            2. 定价模式: 为每个收入来源提供定价策略（固定定价、动态定价、订阅制等）。
            3. 支付方式: 说明客户将如何付款，以及你如何收取费用。
            4. 收入构成: 估算每个收入来源占总收入的百分比。
            5. 利润率: 评估每个收入流的预期利润率。
            6. 可持续性: 分析每个收入流的长期可持续性和增长潜力。
            7. 季节性: 讨论收入是否存在季节性波动，如何应对。
            提供一个全面的收入模型，确保业务的财务可行性和可持续性。
            """,

            "key_resources": f"""
            确定{industry}行业的{target_market}市场所需的关键资源:
            1. 物理资源: 列出所需的主要设备、设施、原材料等。
            2. 智力资源: 描述所需的专利、版权、数据、算法等。
            3. 人力资源: 详述所需的关键人才和技能。
            4. 金融资源: 估算所需的初始资金和运营资金。
            5. 品牌资源: 说明品牌对业务的重要性和如何建立。
            6. 技术资源: 列出关键的技术平台或系统。
            7. 合作伙伴资源: 描述可能需要的关键合作伙伴资源。
            对每种资源，评估其获取难度、成本和对业务成功的重要性。
            """,

            "key_activities": f"""
            列出{industry}行业的{target_market}市场中的关键活动:
            1. 生产活动: 详述产品开发或服务提供的核心流程。
            2. 问题解决: 说明如何解决客户的具体问题或需求。
            3. 平台/网络: 如果适用，描述平台维护和发展的活动。
            4. 供应链管理: 解释原材料或服务采购的关键活动。
            5. 营销和销售: 详述客户获取和维系的主要活动。
            6. 研发: 描述持续创新和改进的活动。
            7. 客户支持: 说明如何提供卓越的客户服务。
            对每个活动，评估其对价值主张的贡献、所需资源和执行挑战。
            """,

            "key_partnerships": f"""
            为{industry}行业的{target_market}市场确定关键合作伙伴:
            1. 战略联盟: 列出可能的非竞争对手合作伙伴。
            2. 供应商: 确定关键供应商及其重要性。
            3. 合资企业: 探讨可能的合资机会。
            4. 买方-供应商关系: 描述与关键客户的深度合作。
            5. 技术合作: 列出可能的技术提供商或集成商。
            6. 分销合作: 确定可能的分销渠道合作伙伴。
            7. 研发合作: 探讨可能的研究机构或大学合作。
            对每个合作关系，说明其目的、预期收益和潜在风险。提供建立和维护这些合作关系的策略。
            """,

            "cost_structure": f"""
            分析{industry}行业的{target_market}市场的成本结构:
            1. 固定成本: 列出并估算主要的固定成本。
            2. 可变成本: 确定主要的可变成本及其与销量的关系。
            3. 规模经济: 讨论如何通过扩大规模降低成本。
            4. 范围经济: 探讨如何通过多元化降低成本。
            5. 成本驱动: 确定主要的成本驱动因素。
            6. 价值驱动: 说明哪些成本对创造价值最为重要。
            7. 盈亏平衡点: 估算达到盈亏平衡所需的销量或收入。
            8. 成本优化: 提出可能的成本优化策略。
            提供一个详细的成本模型，确保业务模式的财务可行性。
            """
        }

        return prompts


    def evaluate_business_model(self, model: Dict[str, str]) -> Dict[str, Any]:
        evaluation_prompt = f"""
        请严格评估以下商业模式，考虑以下方面:
        1. 一致性: 各要素之间是否相互支持和协调?
        2. 可行性: 模式在现实世界中是否可行?需要哪些关键资源和能力?
        3. 盈利能力: 收入模式是否可持续?成本结构是否合理?
        4. 市场吸引力: 目标市场的规模和增长潜力如何?
        5. 竞争优势: 价值主张是否足够强大以区别于竞争对手?
        6. 可扩展性: 模式是否易于扩展到更大的市场或不同地区?
        7. 风险评估: 主要风险和潜在障碍是什么?

        对每个方面进行1-10的评分，并提供详细解释。
        最后，给出总体评分和改进建议。

        商业模式:
        {json.dumps(model, indent=2)}

        请以JSON格式返回评估结果，包括每个方面的评分和解释，以及总体评分和建议。
        """

        evaluation_result = self.llm_processor.process_llm_request(evaluation_prompt)
        return json.loads(evaluation_result)

    def optimize_business_model(self, model: Dict[str, str], evaluation: Dict[str, Any]) -> Dict[str, str]:
        optimization_prompt = f"""
        基于以下评估，优化商业模式:
        {json.dumps(evaluation, indent=2)}

        原始商业模式:
        {json.dumps(model, indent=2)}

        请提供具体的改进建议，包括:
        1. 如何提高各要素间的一致性
        2. 增强价值主张
        3. 优化成本结构和收入模式
        4. 减轻主要风险
        5. 提高可扩展性

        对每个要素提供详细的优化建议。返回一个优化后的商业模式，格式与原始模式相同。
        """

        optimized_model = self.llm_processor.process_llm_request(optimization_prompt)
        return json.loads(optimized_model)

    def create_optimized_business_model(self, industry: str, target_market: str, iterations: int = 3) -> Dict[str, Any]:
        model = self.generate_business_model(industry, target_market)
        evaluation_history = []

        for i in range(iterations):
            evaluation = self.evaluate_business_model(model)
            evaluation_history.append(evaluation)
            model = self.optimize_business_model(model, evaluation)

        final_evaluation = self.evaluate_business_model(model)

        return {
            "final_model": model,
            "final_evaluation": final_evaluation,
            "evaluation_history": evaluation_history
        }

    def human_review_and_refinement(self, model: Dict[str, str]) -> Dict[str, str]:
        review_prompt = f"""
        请审查以下商业模式，并提供改进建议:
        {json.dumps(model, indent=2)}

        1. 哪些方面需要进一步澄清或详细说明?
        2. 是否有任何关键的市场趋势或技术发展未被考虑?
        3. 商业模式中是否存在任何逻辑漏洞或不一致之处?
        4. 如何进一步提高该商业模式的创新性和竞争力?

        请提供具体的修改建议。
        """

        human_feedback = input(review_prompt + "\n\n请输入您的反馈：")

        refinement_prompt = f"""
        基于以下人类专家的反馈，优化商业模式:
        {human_feedback}

        原始商业模式:
        {json.dumps(model, indent=2)}

        请返回一个优化后的商业模式，格式与原始模式相同。
        """

        refined_model = self.llm_processor.process_llm_request(refinement_prompt)
        return json.loads(refined_model)

    def validate_assumptions(self, model: Dict[str, str]) -> List[Dict[str, Any]]:
        validation_prompt = f"""
        基于以下商业模式，识别关键假设并提出验证方法:
        {json.dumps(model, indent=2)}

        对于每个关键假设，请提供:
        1. 假设描述
        2. 相关的商业模式要素
        3. 验证方法 (如市场调研、客户访谈、小规模试点等)
        4. 成功标准

        请以JSON格式返回一个假设列表，每个假设包含上述四个字段。
        """

        assumptions = self.llm_processor.process_llm_request(validation_prompt)
        return json.loads(assumptions)


if __name__ == "__main__":
    db_path = "llm_results.db"
    openai_base_url = "http://localhost:5000/v1"
    openai_api_key = "810001a0a02948d5bf640a98cb69f653"

    llm_processor = LLMProcessor(db_path, openai_base_url, openai_api_key)

    topic = "银发经济"
    generator = BusinessModelGenerator(llm_processor)
    result = generator.create_optimized_business_model("银发经济", "家具行业", iterations=3)
    final_model = result["final_model"]
    final_evaluation = result["final_evaluation"]

    refined_model = generator.human_review_and_refinement(final_model)
    assumptions = generator.validate_assumptions(refined_model)

    print("最终商业模式:", json.dumps(refined_model, indent=2))
    print("最终评估:", json.dumps(final_evaluation, indent=2))
    print("待验证假设:", json.dumps(assumptions, indent=2))





