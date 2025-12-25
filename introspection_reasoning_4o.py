import json
import os
import random
import re
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
# from utils.llama_utils import generate_text
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

# with open("./output/result_gpt4o_test.json", 'r', encoding='utf-8') as file:
#     inspection_result = json.load(file)



with open("output/result_gpt4o_test1.json", 'r', encoding='utf-8') as file:
    inspection_result = json.load(file)
#
#
# with open("./datasets/same_judge_test1.json", 'r', encoding='utf-8') as file:
#     # 加载JSON文件内容到data变量中
#     same_judge_result = json.load(file)

with open("./output/result_gpt4o_investigation.json", 'r', encoding='utf-8') as file:
    # 加载JSON文件内容到data变量中
    investigation_result = json.load(file)


with open("./datasets/relevant_evidence_w_filter1.json", 'r', encoding='utf-8') as file:
    relevant_evidence = json.load(file)

with open("./datasets/filter_index.json", 'r', encoding='utf-8') as file:
    filter_index = json.load(file)

with open("./datasets/text_summary_llama.json", 'r', encoding='utf-8') as file:
    text_summary = json.load(file)

# matching_indices = [index for index, obj in enumerate(investigation_result) if  obj["investigation_answer"]== "Partially Related" and  (obj['same_judge_answer']=="" or obj['same_judge_answer']=="Not Related")]
# matching_indices = [index for index, obj in enumerate(investigation_result) if  obj['same_judge_answer']=="Related"]
# matching_indices = [index for index, obj in enumerate(investigation_result) if  not (obj['same_judge_answer']=="Related") and not (obj["investigation_answer"]== "Partially Related" and  (obj['same_judge_answer']=="" or obj['same_judge_answer']=="Not Related"))]
#
external_info = [external_info[i] for i in matching_indices]

# investigation_result = [investigation_result[i] for i in matching_indices]
# inspection_result = [inspection_result[i] for i in matching_indices]
relevant_evidence = [relevant_evidence[i] for i in matching_indices]
filter_index = [filter_index[i] for i in matching_indices]
text_summary = [text_summary[i] for i in matching_indices]


with open("./output/result_gpt4o_introspection.json", 'r', encoding='utf-8') as file:
    result_list = json.load(file)
length = len(result_list)

batch_size = 16

for index, (external, inspection, investigation, relevant, filter_idx, summary) in enumerate(
            zip(external_info, inspection_result, investigation_result, relevant_evidence, filter_index, text_summary)):

    print(length+index)
    caption = external['caption']
    gt = external['falsified']
    insp = inspection['result']
    insp_a = inspection['answer']
    inv = investigation['investigation_result']
    evidence = external['evidence'] + external['google_evidence']
    evidence = [x for x in evidence if x is not None]
        # evidence = external['filtered_evidence']
    inv_a = ""
    if investigation['investigation_answer'] == "Related":
        inv_a = 'yes'
    elif investigation['investigation_answer'] == "Not Related":
        inv_a = 'no'
    elif investigation['investigation_answer'] == "":
        inv_a = insp_a.rstrip('.').lower()

    text_list = [t['text'] for t in external['external_info'] if 'text' in t and len(t['text']) > 150] + [t['text']
                                                                                                              for
                                                                                                              t in
                                                                                                              external[
                                                                                                                  'google_external_info']
                                                                                                              if len(
                t['text']) > 150]

    # final_text = ""
    # t_list = []
    #     for tx in text_list:
    #         if tx in t_list:
    #             continue
    #         t_list.append(tx)
    #     f_list = []
    #     exclude_list = []
    #     if len(filter_idx) != 0:
    #         for id in filter_idx:
    #             if id is None or id <= 0 or id > len(t_list):
    #                 while True:
    #                     num = random.randrange(0, len(t_list))
    #                     if num not in exclude_list:
    #                         break
    #                 f_list.append(t_list[num])
    #                 exclude_list.append(num)
    #             else:
    #                 f_list.append(t_list[id - 1])
    #                 exclude_list.append(id - 1)
    #
    #     elif len(text_list) != 0:
    #         f_list = t_list[:3]
    #
    #     for i, tx in enumerate(f_list):
    #         # final_text += f"Paragraph {i+1}:\n"
    #         final_text += tx
    #         final_text += "\n\n"
    #     final_text = final_text[:22000]


    if investigation['same_judge_answer'] == 'Related':
        final = "Match"

        prompt = f'''You are an expert in evaluating whether an image and a given caption match.
            Caption: {caption}
            You retrieved additional information relevant to the image:
            Golden retrieved information (Title/Image Caption):{';'.join('%s' %a for a in evidence)}
            Your final decision is: The image and the caption are {final}, the reason is: {investigation['same_judge_result']}
            Based on the above information, please give a detailed reason of your final decision.
            You should only respond in format as described below. DO NOT RETURN ANYTHING ELSE. START YOUR RESPOND WITH '{{'.
            {{"Judgment": "Match / Mismatch","Reason": ""Reason for your judgement""}}
            Response:
            '''

        message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt}
                ],
            )
    else:

        if investigation['investigation_answer'] == "Partially Related":
            prompt = f'''You are an expert in evaluating whether an image and a given caption match.
                Caption: {caption}
                1.Based on the understanding of the image, you provided a consistency check result: {insp}.
                2.You retrieved additional information relevant to the image:
                Golden retrieved information (Title/Image Caption):{';'.join('%s' %a for a in evidence)}
                Summarization of the web pages:{summary}
                Since the additional information is partially related to the caption, please consider the results of the consistency check and the relationship between the additional information and the caption to determine the final result.
                Do not assume a mismatch simply because of the presence of unrelated information.
                You should only respond in format as described below. DO NOT RETURN ANYTHING ELSE. START YOUR RESPOND WITH '{{'.
                {{"Judgment": "Match / Mismatch","Reason": ""Reason for your judgement""}}
                Response:'''
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt}
                ],
            )
        else:
            final = "Match" if inv_a == "yes" else "Mismatch"
            prompt = f'''You are an expert in evaluating whether an image and a given caption match.
            Caption: {caption}
            1.Based on the understanding of the image, you provided a consistency check result: {insp}.
            2.You retrieved additional information relevant to the image:
            Golden retrieved information (Title/Image Caption):{';'.join('%s' %a for a in evidence)}
            Web retrieved relevant information (Possibly related information in the web page):{relevant}
            Your final decision is: The image and the caption are {final}
            Based on the above information, please give a detailed reason of your final decision.
            Do not assume a mismatch simply because of the presence of unrelated information.
            You should only respond in format as described below. DO NOT RETURN ANYTHING ELSE. START YOUR RESPOND WITH '{{'.
            {{"Judgment": "Match / Mismatch","Reason": ""Reason for your judgement""}}
            Response:
            '''
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt}
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
            for res, external in zip(result, [external]):
                res = re.search("{.*?}", res, re.DOTALL).group(0)
                res = res[:55] + res[55:-3].replace('"', '\\"') + res[-3:]
                final = json.loads(res)
                finals.append(final)
            # print(final)
        except Exception as e:
            print(e)
            continue
        break

    for invest, final in zip([investigation], finals):
        invest['introspection_result'] = final['Reason']
        invest['introspection_answer'] = final['Judgment']
        result_list.append(invest)

    filename = "./output/result_gpt4o_introspection.json"
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            pass  # 创建空文件

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(result_list, file, ensure_ascii=False, indent=4)