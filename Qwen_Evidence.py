import json
import os
import random
import re
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.llama_utils import generate_text
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

with open("./output/result_qwen_test.json", 'r', encoding='utf-8') as file:
    inspection_result = json.load(file)


with open("./datasets/relevant_evidence_w_filter1.json", 'r', encoding='utf-8') as file:
    relevant_evidence = json.load(file)

with open("./datasets/filter_index.json", 'r', encoding='utf-8') as file:
    filter_index = json.load(file)

with open("./datasets/text_summary_llama.json", 'r', encoding='utf-8') as file:
    text_summary = json.load(file)



with open("./output/result_qwen_evidence1.json", 'r', encoding='utf-8') as file:
    result_list = json.load(file)

external_info = [external_info[i] for i in matching_indices]

relevant_evidence = [relevant_evidence[i] for i in matching_indices]
filter_index = [filter_index[i] for i in matching_indices]
text_summary = [text_summary[i] for i in matching_indices]


length = len(result_list)

batch_size = 16

for index, (external, inspection, relevant, filter_idx, summary) in enumerate(
            zip(external_info, inspection_result, relevant_evidence,filter_index,text_summary)):
    print(index+length)
    caption = external['caption']
    gt = external['falsified']
    insp = inspection['result']
    insp_a = inspection['answer']
    evidence = external['evidence'] + external['google_evidence']
    evidence = [x for x in evidence if x is not None]
    # evidence = external['filtered_evidence']

    if len(evidence) != 0:

        text_list = [t['text'] for t in external['external_info'] if 'text' in t and len(t['text']) > 150] + [
            t['text']
            for
            t in
            external[
                'google_external_info']
            if len(
                t['text']) > 150]

        selected_items = len(text_list)-1
        final_text = text_list[-1] if len(text_list)!=0 else ""
        prompt = f'''You are an expert in evaluating whether an image and a given caption match.
        Caption: {caption}
        You retrieved additional information relevant to the image:
        Golden retrieved information (Title/Image Caption):{';'.join('%s' % a for a in evidence)}
        Web retrieved relevant information (Possibly related information in the web page):{final_text}
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


    finals = []
    final = ''
    for retry in range(0, 10):
        try:
            result = generate_text(message)
            result = [result]
            print(result)
            for res, external in zip(result, [external]):
                if len(evidence)!=0 or summary!="":
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

    for invest, final in zip([inspection], finals):
        invest['evidence_result'] = final['Reason']
        invest['evidence_answer'] = final['Judgment']
        result_list.append(invest)

    filename = "./output/result_qwen_evidence1.json"
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            pass  # 创建空文件

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(result_list, file, ensure_ascii=False, indent=4)