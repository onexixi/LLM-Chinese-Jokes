import os
from datetime import datetime

from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

QUESTION_VECTOR_PATH= '../tmp'
example_few_show=[]

rcg_question = """
 你是一个专业的用户真实问题识别机器人
 请参考下面的例子识别用户问题的类型
-请只输出 问题和是否预设问题的结果
-禁止解释

 用户问题：{question}

 <example>
 question	真实问题	是否预设问题	
 {example}
 </example>


 用户问题：{question}
请使用如下json输出结果：
-real_question 用户真实问题
-preset 是/否 预设问题
 ```json
{{
    "real_question": ""
    "preset":""
}}
"""

def get_vector_db(QUESTION_VECTOR_PATH=None, example_few_show=None):
    embeddings = HuggingFaceEmbeddings(model_name='D:\\opt\\deployments\\modle\\bge-base-zh-v1.5',
                                       model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    vector_path = QUESTION_VECTOR_PATH + '/ck-qa-' + datetime.today().strftime('%Y%m%d')
    if os.path.exists(vector_path):
        db = FAISS.load_local(vector_path, embeddings)
    else:
        few_shot_docs = [
            Document(page_content=question.split('\t')[0], metadata={'real_question': question.split('\t')[1],"preset":question.split('\t')[2]}) for
            question in
            example_few_show]
        print(few_shot_docs)
        db = FAISS.from_documents(few_shot_docs, embeddings)
        db.save_local(vector_path)
    return db


def get_llm_result(token, prompt, temperature, max_tokens):
    pass


def parse_json_markdown(result):
    pass


@retry(stop_max_attempt_number=2, wait_fixed=1000)
def get_user_reg(db, token, input, rcg_question=None):
    try:
        result_documents = db.similarity_search(input, k=20, fetch_k=5)
        prompt = rcg_question.format(question=input, example=result_documents)
        result = get_llm_result(token=token, prompt=prompt, temperature=0.2, max_tokens=512)
        print(input + ":" + result)
        json_data = parse_json_markdown(result)
        return json_data['real_question'],json_data['preset']
    except Exception as e:
        print(f"Error parsing JSON for {str(e)}")
    return "",""


if __name__ == '__main__':
    import pandas as pd
    import concurrent.futures

    # Read the log0705.xlsx file
    data = pd.read_excel("log0705.xlsx")


    db = get_vector_db()


    def process_input(input):
        result = get_user_reg(db, input)
        return [result[0], result[1]]  # modify this line to return a list or tuple of the desired values


    # Create a thread pool with a specified number of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Apply get_user_reg() to each element in the 5th column using the thread pool
        results = list(executor.map(process_input, data.iloc[:, 4]))

    # Store the results in two new columns of the DataFrame
    data.insert(len(data.columns), 'Column6', [item[0] for item in results])
    data.insert(len(data.columns), 'Column7', [item[1] for item in results])

    # Save the modified data to a new Excel file
    data.to_excel("log0705_modified.xlsx", index=False)