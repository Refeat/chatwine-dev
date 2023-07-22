import os
import configparser

config = configparser.ConfigParser()
config.read('./secrets.ini')
openai_api_key = config['OPENAI']['OPENAI_API_KEY']
serper_api_key = config['SERPER']['SERPER_API_KEY']
serp_api_key = config['SERPAPI']['SERPAPI_API_KEY']
kakao_api_key = config['KAKAO_MAP']['KAKAO_API_KEY']

os.environ.update({'OPENAI_API_KEY': openai_api_key})
os.environ.update({'SERPER_API_KEY': serper_api_key})
os.environ.update({'SERPAPI_API_KEY': serp_api_key})
os.environ.update({'KAKAO_API_KEY': kakao_api_key})

from typing import List, Union, Optional, Any, Dict, cast
import re
import sys
import time
import json
import asyncio
import aiohttp
import requests
import threading

import pandas as pd
from langchain import SerpAPIWrapper, LLMChain
from langchain.agents import Tool, AgentType, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.callbacks.manager import Callbacks
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chains.query_constructor.ir import StructuredQuery
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.document_loaders import DataFrameLoader, SeleniumURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts import PromptTemplate, StringPromptTemplate, load_prompt, BaseChatPromptTemplate
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.schema import AgentAction, AgentFinish, HumanMessage, Document
from langchain.vectorstores import Chroma

import gradio as gr

from tools import WineBarDatabaseTool, WineDatabaseTool, KakaoMapTool, SearchTool
from assistant import Assistant
from user_response_generator import UserResponseGenerator
from agent import Agent


tools = [
    KakaoMapTool(),
    WineBarDatabaseTool(),
    WineDatabaseTool(),
    SearchTool(),
    ]

verbose = True
assistant = Assistant(verbose=verbose)
user_response_generator = UserResponseGenerator(verbose=verbose)
agent = Agent(tools=tools, verbose=verbose)


# hf_writer = gr.HuggingFaceDatasetSaver(huggingface_token, "chatwine-korean")


with gr.Blocks(css='#chatbot .overflow-y-auto{height:750px}') as demo:
    
    with gr.Row():
        gr.HTML("""<div style="text-align: center; max-width: 500px; margin: 0 auto;">
            <div>
                <h1>ChatWine</h1>
            </div>
            <p style="margin-bottom: 10px; font-size: 94%">
                LinkedIn <a href="https://www.linkedin.com/company/audrey-ai/about/">Audrey.ai</a>
            </p>
        </div>""")
    
    chatbot = gr.Chatbot()

    with gr.Row():
        with gr.Column(scale=0.85):
            msg = gr.Textbox()
        with gr.Column(scale=0.15, min_width=0):
            submit_btn = gr.Button("전송")

    user_response_examples = gr.Dataset(samples=[["이번 주에 친구들과 모임이 있는데, 훌륭한 와인 한 병을 추천해줄래?"], ["입문자에게 좋은 와인을 추천해줄래?"], ["연인과 가기 좋은 와인바를 알려줘"]], components=[msg], type="index")
    clear_btn = gr.ClearButton([msg, chatbot])

    dev_mod = True
    cur_stage = gr.Textbox(visible=dev_mod, interactive=False, label='current_stage')
    stage_hist = gr.Textbox(visible=dev_mod, value="stage history: ", interactive=False, label='stage history')
    chat_hist = gr.Textbox(visible=dev_mod, interactive=False, label='chatting_history')
    response_examples_text = gr.Textbox(visible=dev_mod, interactive=False, value="이번 주에 친구들과 모임이 있는데, 훌륭한 와인 한 병을 추천해줄래?|입문자에게 좋은 와인을 추천해줄래?|연인과 가기 좋은 와인바를 알려줘", label='response_examples')
    # btn = gr.Button("Flag", visible=dev_mod)
    # hf_writer.setup(components=[chat_hist, stage_hist, response_examples_text], flagging_dir="chatwine-korean")

    def click_flag_btn(*args):
        # hf_writer.flag(flag_data=[*args])
        pass

    def clean(*args):
        return gr.Dataset.update(samples=[["이번 주에 친구들과 모임이 있는데, 훌륭한 와인 한 병을 추천해줄래?"], ["입문자에게 좋은 와인을 추천해줄래?"], ["연인과 가기 좋은 와인바를 알려줘"]]), "", "stage history: ", "", "이번 주에 친구들과 모임이 있는데, 훌륭한 와인 한 병을 추천해줄래?|입문자에게 좋은 와인을 추천해줄래?|연인과 가기 좋은 와인바를 알려줘"

    def load_example(response_text, input_idx):
        response_examples = []
        for user_response_example in response_text.split('|'):
            response_examples.append([user_response_example])
        return response_examples[input_idx][0]

    async def agent_run(agent_exec, inp, sender):
        sender[0] = ""
        await agent_exec.arun(inp)

    def user_chat(user_message, chat_history_list, chat_history):
        return (chat_history_list + [[user_message, None]], chat_history + f"User: {user_message} <END_OF_TURN>\n", [])

    async def bot_stage_pred(user_response, chat_history, stage_history):
        pre_chat_history = '<END_OF_TURN>'.join(chat_history.split('<END_OF_TURN>')[:-2])
        if pre_chat_history != '':
            pre_chat_history += '<END_OF_TURN>'
        # stage_number = unified_chain.stage_analyzer_chain.run({'conversation_history': pre_chat_history, 'stage_history': stage_history.replace('stage history: ', ''), 'last_user_saying':user_response+' <END_OF_TURN>\n'})
        stage_number = await assistant.arun(conversation_history=pre_chat_history, stage_history= stage_history.replace('stage history: ', ''), last_user_saying=user_response+' <END_OF_TURN>\n')
        stage_number = stage_number[-1]
        stage_history += stage_number if stage_history == "stage history: " else ", " + stage_number

        return stage_number, stage_history

    async def bot_chat(user_response, chat_history, chat_history_list, current_stage): # stream output by yielding
        
        pre_chat_history = '<END_OF_TURN>'.join(chat_history.split('<END_OF_TURN>')[:-2])
        if pre_chat_history != '':
            pre_chat_history += '<END_OF_TURN>'

        sender = ["", False]
        task = asyncio.create_task(agent.arun(sender = sender, input=user_response+' <END_OF_TURN>\n', conversation_history=pre_chat_history, stage_number= current_stage))
        await asyncio.sleep(0)
        while(sender[1] == False):
            await asyncio.sleep(0.2)
            chat_history_list[-1][1] = sender[0]
            yield chat_history_list, chat_history + f"이우선: {sender[0]}<END_OF_TURN>\n"
        # resp = agent.run(sender = sender, input=user_response+' <END_OF_TURN>\n', conversation_history=pre_chat_history, stage_number= current_stage)

        chat_history_list[-1][1] = sender[0]
        # chat_history_list[-1][1] = resp
        yield chat_history_list, chat_history + f"이우선: {sender[0]}<END_OF_TURN>\n"

    async def bot_response_pred(chat_history):
        response_examples = []
        pre_chat_history = '<END_OF_TURN>'.join(chat_history.split('<END_OF_TURN>')[-3:])
        out = await user_response_generator.arun(conversation_history=pre_chat_history)
        for user_response_example in out.split('|'):
            response_examples.append([user_response_example])
        return [response_examples, out, ""]
    
    # btn.click(lambda *args: hf_writer.flag(args), [msg, chat_hist, stage_hist, response_examples_text], None, preprocess=False)

    msg.submit(
        user_chat, [msg, chatbot, chat_hist], [chatbot, chat_hist, user_response_examples], queue=False
    ).then(
        bot_stage_pred, [msg, chat_hist, stage_hist], [cur_stage, stage_hist], queue=False
    ).then(
        bot_chat, [msg, chat_hist, chatbot, cur_stage], [chatbot, chat_hist]
    ).then(
        bot_response_pred, chat_hist, [user_response_examples, response_examples_text, msg]
    ).then(
        click_flag_btn, [chat_hist, stage_hist, response_examples_text], None
    )

    submit_btn.click(
        user_chat, [msg, chatbot, chat_hist], [chatbot, chat_hist, user_response_examples], queue=False
    ).then(
        bot_stage_pred, [msg, chat_hist, stage_hist], [cur_stage, stage_hist], queue=False
    ).then(
        bot_chat, [msg, chat_hist, chatbot, cur_stage], [chatbot, chat_hist]
    ).then(
        bot_response_pred, chat_hist, [user_response_examples, response_examples_text, msg]
    ).then(
        click_flag_btn, [chat_hist, stage_hist, response_examples_text], None
    )

    clear_btn.click(
        clean, 
        inputs=[user_response_examples, cur_stage, stage_hist, chat_hist, response_examples_text], 
        outputs=[user_response_examples, cur_stage, stage_hist, chat_hist, response_examples_text], 
        queue=False)
    user_response_examples.click(load_example, inputs=[response_examples_text, user_response_examples], outputs=[msg], queue=False)
    # btn.click(lambda *args: hf_writer.flag(args), [chat_hist, stage_hist, response_examples_text], None, preprocess=False)
demo.queue(concurrency_count=100)
demo.launch(server_name='0.0.0.0', server_port=9441)
