import json
import os

from tqdm import tqdm

from utils.llama_utils import generate_text


class MemoryState:

    initial_query:str = ''
    inspection_result: str = ''
    inspection_answer: str = ''
    same_judge_result: str = ''
    same_judge_answer: str = ''
    investigation_result: str = ''
    investigation_answer: str = ''

    def __init__(self, initial_query=''):
        self.initial_query = initial_query
import json



with open("./datasets/VERITE_test_external_info_final1.json", 'r', encoding='utf-8') as file:
    external_info = json.load(file)

with open("./output/2/inspection_reasoning29_VERITE.json", 'r', encoding='utf-8') as file:
    inspection_result = json.load(file)


with open("./output/2/same_judge_reasoning_VERITE.json", 'r', encoding='utf-8') as file:
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

            message = [
                {"role": "user", "content": prompt}
            ]
            for retry in range(0, 10):
                try:
                    output = generate_text(message)

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

    filename = "./output/2/same_judge_reasoning_VERITE.json"
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            pass  # 创建空文件

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(result_list, file, ensure_ascii=False, indent=4)