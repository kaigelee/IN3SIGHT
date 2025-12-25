import json
import os
import random
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

with open("./output/2/inspection_reasoning1.json", 'r', encoding='utf-8') as file:
    inspection_result = json.load(file)


with open("./datasets/text_summary_llama.json", 'r', encoding='utf-8') as file:
    text_summary = json.load(file)

with open("./datasets/relevant_evidence_w_filter1.json", 'r', encoding='utf-8') as file:
    relevant_evidence = json.load(file)



with open("./output/result_baseline_no_aug_summary.json", 'r', encoding='utf-8') as file:
    result_list = json.load(file)
length = len(result_list)

batch_size = 16
for batch_start in range(length, len(external_info), batch_size):
    # 获取4个连续样本
    external_batch = external_info[batch_start:batch_start + batch_size]
    inspection_batch = inspection_result[batch_start:batch_start + batch_size]


    summary_batch = text_summary[batch_start:batch_start + batch_size]
    relevant_batch = relevant_evidence[batch_start:batch_start+batch_size]
    messages = []
    for index, (external,inspection,summary,relevant) in enumerate(
            zip(external_batch, inspection_batch, summary_batch, relevant_batch)):
        sample_index = batch_start + index
        print(sample_index)
        caption = external['caption']
        gt = external['falsified']
        # evidence_text = external['detailed_evidence']
        judgment_content = {}
        evidence = external['evidence'] + external['google_evidence']
        evidence = [x for x in evidence if x is not None]
        # evidence = external['evidence']+external['google_evidence']
        # evidence = external['filtered_evidence']
        if len(evidence)!=0:

            text_list = [t['text'] for t in external['external_info'] if 'text' in t and len(t['text']) > 150] + [
                t['text']
                for
                t in
                external[
                    'google_external_info']
                if len(
                    t['text']) > 150]

            selected_items = random.sample(text_list, 3 if len(text_list) > 3 else len(text_list))
            final_text = '\n\n'.join(selected_items)
            final_text = final_text[:22000]
            prompt = f'''You are an expert in evaluating whether an image and a given caption match.
            Caption: {caption}
            You retrieved additional information relevant to the image:
            Golden retrieved information (Title/Image Caption):{';'.join('%s' % a for a in evidence)}
            Web retrieved relevant information (Possibly related information in the web page):{summary}
            Based on the above information, please give a detailed reason of your final decision.
            Do not assume a mismatch simply because of the presence of unrelated information.
            You should only respond in format as described below. DO NOT RETURN ANYTHING ELSE. START YOUR RESPOND WITH '{{'.
            {{"Judgment": "Match / Mismatch","Reason": ""Reason for your judgement""}}
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
                if len(external['evidence'] + external['google_evidence'])!=0 or summary!="":
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

    for inspection,final in zip(inspection_batch,finals):
        inspection['investigation_result'] = final['Reason']
        inspection['investigation_answer'] = final['Judgment']
        result_list.append(inspection)

    filename = "./output/result_baseline_no_aug_summary.json"
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            pass  # 创建空文件

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(result_list, file, ensure_ascii=False, indent=4)