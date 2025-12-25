import json
import os
import re

from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm


import json


os.environ["OPENAI_API_KEY"] = "sk-itzvOjtKAeuWdEojgC2cO74vbVwiSgRMhVj9iZrU6magTBib"

with open("./datasets/output_data_test_400.json", 'r', encoding='utf-8') as file:
    test_samples = json.load(file)

with open("./datasets/test_external_info_llama_final.json", 'r', encoding='utf-8') as file:
    external_info = json.load(file)

matching_indices = []
for ins in test_samples:
    id = ins['id']
    for i,item in enumerate(external_info):
        if item['id']==id:
            matching_indices.append(i)
            break


with open("./output/result_gpt4o_same_judge.json", 'r', encoding='utf-8') as file:
    same_judge_result = json.load(file)


with open("./datasets/text_summary_llama.json", 'r', encoding='utf-8') as file:
    text_summary = json.load(file)

with open("./datasets/relevant_evidence_w_filter1.json", 'r', encoding='utf-8') as file:
    relevant_evidence = json.load(file)

external_info = [external_info[i] for i in matching_indices]

text_summary = [text_summary[i] for i in matching_indices]
relevant_evidence = [relevant_evidence[i] for i in matching_indices]


with open("./output/result_gpt4o_investigation.json", 'r', encoding='utf-8') as file:
    result_list = json.load(file)
length = len(result_list)

batch_size = 16

for index, (external,same,summary,relevant) in enumerate(
            zip(external_info[length:], same_judge_result[length:], text_summary[length:], relevant_evidence[length:])):
    print(index+length)
    caption = external['caption']
    gt = external['falsified']
    inspection_result = same['inspection_result']
        # evidence_text = external['detailed_evidence']
    judgment_content = {}
    evidence = external['evidence'] + external['google_evidence']
    evidence = [x for x in evidence if x is not None]
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

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt}
            ],
        )
    else:
        message = HumanMessage(
                content=[
                    {"type": "text", "text": "just output an empty string"}
                ],
            )

    finals = []
    final = ''
    for retry in range(0, 10):
        try:
            model = ChatOpenAI(model="gpt-4o", openai_api_base="https://api.agicto.cn/v1")
            response = model.invoke([message])
            result = response.content
            result = [result]
            print(result)
            for res,external,summary in zip(result,[external],[summary]):
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

    for same,final in zip([same],finals):
        same['investigation_result'] = final['Reason']
        same['investigation_answer'] = final['Judgment']
        result_list.append(same)

    filename = "./output/result_gpt4o_investigation.json"
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            pass  # 创建空文件

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(result_list, file, ensure_ascii=False, indent=4)