import asyncio
import aiohttp
import json

API_KEY = "your_api_key_here"
API_URL = "https://api.openai.com/v1/chat/completions"


async def query_llm(prompt):
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}]
        }
        async with session.post(API_URL, headers=headers, data=json.dumps(data)) as response:
            result = await response.json()
            return result['choices'][0]['message']['content']


async def execute_step(step_name, question, rules):
    prompt = f"""
    步骤: {step_name}
    问题: {question}
    应用以下规则:
    {', '.join(rules)}

    请根据上述规则，执行这个步骤并给出具体建议。
    """
    return await query_llm(prompt)


async def problem_solving_framework(question):
    steps = [
        {
            "name": "步骤1：识别和定义问题",
            "rules": ["黑匣子思维(4)", "能力圈(14)", "关注因素(30)", "关注假设(31)", "主动思考(32)"]
        },
        {
            "name": "步骤2：调整心态",
            "rules": ["心理账户(1)", "期望管理(49)", "对谦虚的赞美(51)", "内心的成功(52)", "无知的优势(53)",
                      "独立思考(54)"]
        },
        {
            "name": "步骤3：收集信息和分析",
            "rules": ["史特金定律(50)", "站在对方立场上思考(40)", "避免教条的陷阱(37)", "避免货物崇拜(44)",
                      "信息的力量(3)", "数据的局限性(13)", "区分证据和假设(29)"]
        },
        {
            "name": "步骤4：制定策略",
            "rules": ["修正的巨大作用(2)", "治病不如防病(33)", "考虑适得其反(5)", "简单胜于复杂(7)", "边际效应(17)",
                      "目标置换(41)", "费用置换(42)"]
        },
        {
            "name": "步骤5：执行",
            "rules": ["坚持的秘密(15)", "专注力陷阱(35)", "思考极点(39)", "选择的成本(18)", "努力清单(45)",
                      "讲好故事(46)", "保持一致性(47)"]
        },
        {
            "name": "步骤6：评估和调整",
            "rules": ["思维减法(38)", "相关性原则(6)", "滞后效应(8)", "系统思考(10)", "规模思维(11)", "参照系(12)",
                      "概率思维(22)"]
        },
        {
            "name": "步骤7：反思和学习",
            "rules": ["记忆账户(21)", "两个自我(20)", "自我关怀(23)", "定义问题的艺术(9)", "经验的陷阱(36)",
                      "分析错误(43)", "放大镜效应(48)"]
        }
    ]

    overall_principles = [
        "美好生活的消极艺术(6)",
        "快乐论与实现论(25)",
        "尊严圈(26-28)",
        "拥抱矛盾(16)",
        "合作博弈(19)",
        "信息的价值(24)"
    ]

    # 执行所有步骤
    step_tasks = [execute_step(step["name"], question, step["rules"]) for step in steps]
    step_results = await asyncio.gather(*step_tasks)

    # 应用总体原则
    overall_prompt = f"""
    问题: {question}
    考虑以下总体原则:
    {', '.join(overall_principles)}

    请根据这些总体原则，对整个问题解决过程提供指导和建议。
    """
    overall_guidance = await query_llm(overall_prompt)

    return list(zip([step["name"] for step in steps], step_results)), overall_guidance


async def main():
    question = "如何在工作中更好地管理时间和提高效率？"

    print(f"问题: {question}\n")
    results, overall_guidance = await problem_solving_framework(question)

    for step_name, result in results:
        print(f"{step_name}:")
        print(f"{result}\n")

    print("总体指导:")
    print(overall_guidance)


if __name__ == "__main__":
    asyncio.run(main())