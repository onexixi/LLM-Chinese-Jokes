import asyncio
from openai import OpenAI
from prompt.though_prompt import rule_list, do_rul_prompt

API_KEY = "lm-studio"
API_URL = "http://localhost:5000/v1"

client = OpenAI(base_url="http://localhost:5000/v1", api_key="810001a0a02948d5bf640a98cb69f653")


def query_llm(prompt):
    completion = client.chat.completions.create(
        model="gemma-2/27b-it-IMat-GGUF",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content


async def execute_step(step_name, question, prompt):
    formatted_prompt = do_rul_prompt.format(question=question, step=step_name, rul_prompt=prompt)
    result = query_llm(formatted_prompt)
    print(f"{formatted_prompt}:\n{result}\n")
    return result


async def apply_overall_principles(question, principles):
    result_tmp = ""
    for step_name, method_results in principles:
        for result in method_results:
            result_tmp += result + "\n"

    prompt = f"""
    问题: {question}
    -请使用中文回答
    -请按照步骤总结核心观念
    --分析内容：
    {result_tmp}
    请根据这些总体原则，对整个问题解决过程提供指导和建议。
    """
    return query_llm(prompt)


def get_rul(rul):
    matching_rules = [rule for rule in rule_list if rul in rule]
    return matching_rules[0] if matching_rules else f"未找到匹配的规则: {rul}"


async def problem_solving_framework(question):
    steps = [
        {
            "name": "步骤1：识别和定义问题",
            "methods": ["黑匣子思维", "能力圈"]
        },
        {
            "name": "步骤2：调整心态",
            "methods": ["心理账户", "期望管理", "对谦虚之美的赞美"]
        },
        {
            "name": "步骤3：收集信息和分析",
            "methods": ["史特金定律", "角色调换", "避免教条的陷阱", "摆脱货物崇拜"]
        },
        {
            "name": "步骤4：制定策略",
            "methods": ["修正的巨大作用", "消极的艺术", "适得其反效应"]
        },
        {
            "name": "步骤5：执行",
            "methods": ["坚持的秘密", "专注力陷阱", "思考极点"]
        },
        {
            "name": "步骤6：评估和调整",
            "methods": ["内心的成功", "思维减法"]
        },
        {
            "name": "步骤7：反思和学习",
            "methods": ["记忆账户", "两个自我"]
        }]

    overall_principles = [
        "保持'美好生活的消极艺术'的态度，避免重大错误",
        "运用'快乐论与实现论'来平衡幸福感和意义感",
        "使用'尊严圈'来坚持核心原则"
    ]

    step_results = []
    for step in steps:
        step_name = step["name"]
        method_results = await asyncio.gather(
            *(execute_step(step_name, question, get_rul(rul)) for rul in step["methods"]))
        step_results.append((step_name, method_results))

    overall_guidance = await apply_overall_principles(question, step_results)

    return step_results, overall_guidance


async def main():
    question = "在大城市买房，家里人不同意 怎么办"
    print(f"问题: {question}\n")

    results, overall_guidance = await problem_solving_framework(question)

    # 将结果保存到文本文件
    with open("problem_solving_results.txt", "w", encoding="utf-8") as f:
        f.write(f"问题: {question}\n\n")

        for step_name, method_results in results:
            f.write(f"步骤名称: {step_name}\n")
            f.write("方法结果:\n")
            for result in method_results:
                f.write(f"{result}\n")
            f.write("\n")

        f.write("总体建议:\n")
        f.write(f"{overall_guidance}\n")

    print("结果已保存到 problem_solving_results.txt 文件中。")


if __name__ == "__main__":
    asyncio.run(main())