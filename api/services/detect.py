import copy
import json
import os
import sys
import requests
import torch
import time

from api.model_protector import ModelProtector
from api.utils import LRUCache, get_model_state_dict
from api.config import EnvVar
from api.schemas.response import Answer
dectypt = os.getenv('IS_ENCRYPT') != 'false'


class Engine:
    def __init__(self) -> None:
        self.model = LRUCache(1)
        self.base_model = LRUCache(2)


    async def batch_predict(
            self,
            model_id,
            prompts,
            initial_prompt,
            temperature=1,
            max_tokens=1024,
            top_p=1,
    ):
        prompts_json = []
        for prompt  in prompts:
            new_records = []
            for record in prompt.records:
                new_records.append(dict(record))
            prompt.records=new_records
            prompt_json=dict(prompt)
            prompts_json.append(dict(prompt_json))


        if "n98" in model_id:
            endpoint = "http://0.0.0.0:8002/omllava/v1/process/batch_infer"
            model_id = "omchat-7b-chat-n98"

        else:
            endpoint = "http://0.0.0.0:8003/omllava/v1/process/batch_infer"
            model_id = "omchat-7b-chat-n114"

        body = {
            "model_id": model_id,
            "prompts": prompts_json,
            "initial_prompt": initial_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }

        temp = requests.post(endpoint, json=body)
        temp = temp.json()
        generated_texts=[]
        for answer in temp.get("answer"):
            generated_texts.append(Answer(**answer))
        return generated_texts



if __name__ == "__main__":
    e = Engine()
    s_t = time.time()
    model = e.load_model(model_id="omchat-llava-qllama-7b-chat-v1-1-finetune_qlora_zh_n67",
                         # "lq_mcqa_0_314",#"omchat-llava-qllama-7b-chat-v1-1-qllama-finetune_zh_n97",#"omchat-llava-vicuna-7b-v1.5-v1-1-finetune_zh_n92",",#
                         resources_prefix="../../../llm_models"
                         )
    print(time.time() - s_t)

    sampling_params = SamplingParams(
        temperature=0.9, max_tokens=512, top_p=1.0, stop=["<|im_end|>"]
    )
    images = []
    texts = []

    res = model.generate(
        prompts=[[{"user": "图片上有什么"}]],
        images=[{"src_type": "url",
                 "image_src": "https://img0.baidu.com/it/u=56109659,3345510515&fm=253&fmt=auto&app=138&f=JPEG?w=889&h=500"}],
        choices=[[]],
        sampling_params=sampling_params,
        initial_prompt="你好",
    )
    generated_texts = []
    for output in res:
        text = output.outputs[0].text
        input_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)
        generated_texts.append(Answer(content=text, input_tokens=input_tokens, output_tokens=output_tokens))
        print(output.prompt)
    print(generated_texts)
    print(time.time() - s_t)
    print("done")
