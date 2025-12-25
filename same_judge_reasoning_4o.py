import json
import os

from tqdm import tqdm

# from utils.llama_utils import generate_text
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

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

with open("./output/1/inspection_reasoning.json", 'r', encoding='utf-8') as file:
    inspection_result = json.load(file)


external_info = [external_info[i] for i in matching_indices]
inspection_result = [inspection_result[i] for i in matching_indices]


with open("./output/result_gpt4o_same_judge.json", 'r', encoding='utf-8') as file:
    result_list = json.load(file)
length = len(result_list)

for index, (external,inspection) in enumerate(zip(external_info[length:],inspection_result[length:])):
    print(index+length)
    caption = external['caption']
    gt = external['falsified']
    # result = []
    judgment_content = {'Judgment':"", 'Reason':""}

    if len(external['caption_search']) != 0:

        text = external['caption_search'][0]['text']
        print(len(text))
        if len(text)>=150:
            prompt = f'''You will be provided with web crawled information and a claim.
            Web retrieved relevant information (Possibly related information in the web page):{text}
            Claim:{caption}
            Based on the relationship between the information and the claim, you need to choose one of the following options:
            1.Related: The basic information of the retrieved evidence is related to the claim
            2.Not Related: The basic information retrieved is not related to the claim
            You should only respond in format as described below. DO NOT RETURN ANYTHING ELSE. START YOUR RESPOND WITH '{{'.
            {{"Judgment": "Related / Not Related","Reason": ""Provide an explanation""}}
            Response:
            '''

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt}
                ],
            )
            for retry in range(0, 10):
                try:
                    model = ChatOpenAI(model="gpt-4o", openai_api_base="https://api.agicto.cn/v1")
                    response = model.invoke([message])
                    output = response.content

                    # match = re.search(r'"Reason": "(.*?)"', text).group(1)
                    # output = output.replace("'", '"')
                    output= output[:52]+output[52:-3].replace('"', '\\"')+output[-3:]
                    print(gt, output)
                    judgment_content = json.loads(output)

                except Exception as e:
                    print(e)
                    continue

                break


    inspection['same_judge_result'] = judgment_content['Reason']
    inspection['same_judge_answer'] = judgment_content['Judgment']

    result_list.append(inspection)

    filename = "./output/result_gpt4o_same_judge.json"
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            pass  # 创建空文件

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(result_list, file, ensure_ascii=False, indent=4)