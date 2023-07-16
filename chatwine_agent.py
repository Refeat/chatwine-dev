import os
import configparser

config = configparser.ConfigParser()
config.read('./secrets.ini')
openai_api_key = config['OPENAI']['OPENAI_API_KEY']
serper_api_key = config['SERPER']['SERPER_API_KEY']
serp_api_key = config['SERPAPI']['SERPAPI_API_KEY']
kakao_api_key = config['KAKAO_MAP']['KAKAO_API_KEY']
huggingface_token = config['HUGGINGFACE']['HUGGINGFACE_TOKEN']

os.environ.update({'OPENAI_API_KEY': openai_api_key})
os.environ.update({'SERPER_API_KEY': serper_api_key})
os.environ.update({'SERPAPI_API_KEY': serp_api_key})
os.environ.update({'KAKAO_API_KEY': kakao_api_key})
os.environ.update({'HUGGINGFACE_TOKEN': huggingface_token})

huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
kakao_api_key = os.getenv('KAKAO_API_KEY')

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