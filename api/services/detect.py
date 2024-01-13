import os
import requests
from api.utils import LRUCache
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



