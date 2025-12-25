```bash
conda create -n I3OOC python=3.10 -y
conda activate I3OOC

pip install -r requirements.txt
pip install flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```



**根目录**主要文件为inspection_reasoning.py、investigation_reasoning.py、Introspection_reasoning.py，其余为消融实验添加的文件

**eval文件夹**中的文件用来打断点查看ACC指标

eval.py  内部检查阶段后的结果

eval_final.py  外部验证后的结果

eval_joint.py   反思后的结果



**dataset文件夹**

包含以output为首的NewsCLIPpings数据集和VERITE测试集

test_external_info_llama_final.json NewsCLIPpings测试集外部证据检索结果

其他文件是一系列证据检索、翻译等脚本文件



**scripts文件夹**

包含提示优化主要脚本文件shallow_optimizer、数据集dataset构建文件Dataset.py、其他对证据过滤和相关性验证的脚本文件



**主要文件说明：**

shallow_optimizer.py：执行提示优化过程

主要变量：

```python
#我的API Key
os.environ["OPENAI_API_KEY"] = "sk-N4xxTy7s4ZhU4pamqcxv4fNIB3FthtXqZDnq93QyHs2G4aBh"

global_train_file #提取的少量训练样本（200条，100+100）
path #输出结果文件夹路径
history_generate_prompts_path #生成的历史提示和分数
test_prompt_file_path #每个iter的最优提示，即作为下一轮的current_prompt

history_list_size_best = 8 #meta prompt中最优提示的数量
few_shot_num = 16 #参考集的大小
batch_size = 8 
prompt_num = 7 #每个iter生成Prompt的数量

dataloader = custom_dataloader(dataset, num_epochs=5) #控制epoch数


val_path #验证集文件路径


```

inspection_reasoning.py：输入Prompt和测试集样本，输出内部检查结果

主要变量：

```python
#测试集文件路径
filename = "./datasets/test_external_info_llama_final.json"
#输出结果路径
output_filename = "./output/2/test.json"
#bs
batch_size = 8
#待测试Prompt
best_prompt = "Assess the following caption for its accuracy and how well it relates to the provided image: {caption}"

```



Investigation_reasoning.py：进行外部调查

主要变量：

```python
external_info #测试集外部证据列表
same_judge_result #文搜图后图片一致性的验证结果
text_summary #总结证据
relevant_evidence #相关性过滤后的证据
result_list #结果输出文件
```

Introspection_reasoning.py：进行反思和最终结果生成

主要变量：

```python
external_info #测试集外部证据列表
inspection_result #内部检查结果
investigation_result #外部验证结果
relevant_evidence #相关性过滤后的证据
filter_index #过滤网站id
text_summary #总结证据
result_list #最终结果
```

