import json
import os
import re

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

# from datasets.Qwen_instruct_util import generate_text
# from datasets.llama_utils import generate_text
from utils.llama_utils_batch import generate_text_batch

import json
# os.environ["OPENAI_API_KEY"] = "sk-N4xxTy7s4ZhU4pamqcxv4fNIB3FthtXqZDnq93QyHs2G4aBh"

# from langchain_community.chat_models import ChatOpenAI

# model = ChatOpenAI(model="gpt-4o-mini", openai_api_base="https://api.agicto.cn/v1")


with open("./datasets/test_external_info_llama_final.json", 'r', encoding='utf-8') as file:
    external_info = json.load(file)

with open("./output/1/same_judge_reasoning.json", 'r', encoding='utf-8') as file:
    same_judge_result = json.load(file)


with open("./datasets/text_summary_llama.json", 'r', encoding='utf-8') as file:
    text_summary = json.load(file)

with open("./datasets/relevant_evidence_w_filter1.json", 'r', encoding='utf-8') as file:
    relevant_evidence = json.load(file)



with open("./output/2/investigation_reasoning1_test.json", 'r', encoding='utf-8') as file:
    result_list = json.load(file)
length = len(result_list)

batch_size = 16
for batch_start in range(length, len(external_info), batch_size):
    # 获取4个连续样本
    external_batch = external_info[batch_start:batch_start + batch_size]
    same_batch = same_judge_result[batch_start:batch_start + batch_size]


    summary_batch = text_summary[batch_start:batch_start + batch_size]
    relevant_batch = relevant_evidence[batch_start:batch_start+batch_size]
    messages = []
    for index, (external,same,summary,relevant) in enumerate(
            zip(external_batch, same_batch, summary_batch, relevant_batch)):
        sample_index = batch_start + index
        print(sample_index)
        caption = external['caption']
        gt = external['falsified']
        inspection_result = same['inspection_result']
        # evidence_text = external['detailed_evidence']
        judgment_content = {}
        evidence = external['evidence'] + external['google_evidence']
        # evidence = external['google_evidence']
        evidence = [x for x in evidence if x is not None]
        # evidence = [x.split('\n')[0] for x in evidence if x is not None]
        # evidence = external['evidence']+external['google_evidence']
        # evidence = external['filtered_evidence']
        if len(evidence)!=0:

            prompt = f'''You will be provided with golden retrieved information, web crawled information and a claim.
            Golden retrieved information (Title/Image Caption):{';'.join('%s' %a for a in evidence)}
            Web retrieved relevant information (Possibly related information in the web page):{relevant}
            Claim:{caption}
            Based on the relationship between the information and the claim, you need to choose one of the following options:
            1.Related: The basic information of the retrieved text is related to the claim
            2.Partially Related: Although there is no information in the given text that directly supports the claim,, the background of retrieved text is related to the claim
            3.Not Related: The basic information retrieved is not related to the claim
            You should only respond in format as described below. DO NOT RETURN ANYTHING ELSE. START YOUR RESPOND WITH '{{'.
            {{"Judgment": "Related / Partially Related / Not Related","Reason": ""Reason for your judgement""}}
            Response:
            '''

            message = [
                {"role": "user", "content": prompt}]
        else:

            message = [{"role": "user", "content": "just output an empty string"}]

        messages.append(message)

    finals = []
    final = ''
    for retry in range(0, 10):
        try:
            result = generate_text_batch(messages)
            print(result)
            for res,external,summary in zip(result,external_batch,summary_batch):
                if len(external['evidence'])!=0 or summary!="":
                    res = re.search("{.*?}", res, re.DOTALL).group(0)
                    res = res[:55] + res[55:-3].replace('"', '\\"') + res[-3:]
                    final = json.loads(res)
                else:
                    final = {"Judgment":"","Reason":"Evidence is NULL"}
                finals.append(final)
            # print(final)
        except Exception as e:
            print(e)
            continue
        break

    for same,final in zip(same_batch,finals):
        same['investigation_result'] = final['Reason']
        same['investigation_answer'] = final['Judgment']
        result_list.append(same)

    filename = "./output/2/investigation_reasoning1_test.json"
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            pass  # 创建空文件

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(result_list, file, ensure_ascii=False, indent=4)