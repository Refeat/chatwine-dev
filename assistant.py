import os
import configparser

import asyncio

from typing import Dict
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import load_prompt

class Assistant:
    def __init__(self, template_path="./templates/assistant_prompt_template.json", verbose=False):
        stage_analyzer_inception_prompt = load_prompt(template_path)
        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.0)
        stage_analyzer_chain = LLMChain(
            llm=llm,
            prompt=stage_analyzer_inception_prompt, 
            verbose=verbose, 
            output_key="stage_number")

        self.stage_analyzer_chain = stage_analyzer_chain

    async def arun(self, *args, **kwargs):
        resp = await self.stage_analyzer_chain.arun(kwargs)
        return resp
    
if __name__ == "__main__":
    assistant = Assistant()