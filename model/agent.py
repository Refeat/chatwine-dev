import os
import configparser

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
from langchain.callbacks.openai_info import MODEL_COST_PER_1K_TOKENS, get_openai_token_cost_for_model, standardize_model_name
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
from langchain.schema import AgentAction, AgentFinish, HumanMessage, Document, LLMResult
from langchain.vectorstores import Chroma

from model.tools import WineBarDatabaseTool, WineDatabaseTool, KakaoMapTool, SearchTool


conversation_stages_dict = {
    "1": "Introduction: Start the conversation by introducing yourself. Maintain politeness, respect, and a professional tone.",
    "2": "Needs Analysis: Identify the customer's needs to make wine recommendations. Note that the wine database tools are not available. You ask about the occasion the customer will enjoy the wine, what they eat with it, and their desired price point. Ask only ONE question at a time.",
    "3": "Checking Price Range: Asking the customer's preferred price point. Again, remember that the tool for this is not available. But if you know the customer's perferences and price range, then search for the three most suitable wines with tool and recommend wines. Each wine recommendation should form of product cards in a list format with a Vivino link, price, rating, wine type, flavor description, and image. Use only wines available in the database for recommendations. If there are no suitable wines in the database, inform the customer. After making a recommendation, inquire whether the customer likes the suggested wine.",
    "4": "Wine Recommendation: Propose the three most suitable wines based on the customer's needs and price range. Before the recommendation, you should have identified the occasion the customer will enjoy the wine, what they will eat with it, and their desired price point. Each wine recommendation should form of product cards in a list format with a Vivino link, price, rating, wine type, flavor description, and image. Use only wines available in the database for recommendations. If there are no suitable wines in the database, inform the customer. After making a recommendation, inquire whether the customer likes the suggested wine.",
    "5": "Sales: If the customer approves of the recommended wine, provide a detailed description. Supply a product card in a list format with a Vivino link, price, rating, wine type, flavor description, and image.",
    "6": "Location Suggestions: Recommend wine bars based on the customer's location and occasion. Before making a recommendation, always use the map tool to find the district of the customer's preferred location. Then use the wine bar database tool to find a suitable wine bar. Provide form of product cards in a list format with the wine bar's name, url, rating, address, menu, opening_hours, holidays, phone, summary, and image with img_urls. Use only wine bars available in the database for recommendations. If there are no suitable wine bars in the database, inform the customer. After making a recommendation, inquire whether the customer likes the suggested wine.",
    "7": "Concluding the Conversation: Respond appropriately to the customer's comments to wrap up the conversation.",
    "8": "Questions and Answers: This stage involves answering customer's inquiries. Use the search tool or wine database tool to provide specific answers where possible. Describe answer as detailed as possible",
    "9": "Other Situations: Use this step when the situation does not fit into any of the steps between 1 and 8."
}

class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Any]   
    
    def format(self, **kwargs) -> str:
        stage_number = kwargs.pop("stage_number")
        kwargs["conversation_stage"] = conversation_stages_dict[stage_number]
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        special_chars = "()'[]{}"
        for action, observation in intermediate_steps:
            thoughts += action.log

            if ('Desired Outcome: ' in action.log) and (('Action: wine_database' in action.log) or ('Action: wine_bar_database' in action.log)):
                regex = r"Desired Outcome:(.*)"
                match = re.search(regex, action.log, re.DOTALL)
                if not match:
                    raise ValueError(f"Could not parse Desired Outcome: `{action.log}`")
                desired_outcome_keys = [key.strip() for key in match.group(1).split(',')]

                pattern = re.compile(r'metadata=\{(.*?)\}')
                matches = pattern.findall(f'{observation}')
                documents = ['{'+f'{match}'+'}' for match in matches]
                
                pattern = re.compile(r"'(\w+)':\s*('[^']+'|\b[^\s,]+\b)")
                output=[]

                for doc in documents:
                    # Extract key-value pairs from the document string
                    matches = pattern.findall(doc)

                    # Convert matches to a dictionary
                    doc_dict = dict(matches)

                    # Create a new dictionary containing only the desired keys
                    item_dict = {}
                    for key in desired_outcome_keys:
                        value = doc_dict.get(key, "")
                        for c in special_chars:
                            value = value.replace(c, "")                        
                        item_dict[key] = value
                    output.append(item_dict)
                
                observation = ','.join([str(i) for i in output])

            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "이우선: " in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("이우선: ")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*?)\n"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # desired_outcome = match.group(3).strip() if match.group(3) else None

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


class CustomStreamingStdOutCallbackHandler(FinalStreamingStdOutCallbackHandler):
    """Callback handler for streaming in agents.
    Only works with agents using LLMs that support streaming.

    The output will be streamed until "<END" is reached.
    """
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0

    def __init__(
        self,
        *,
        answer_prefix_tokens: Optional[List[str]] = None,
        end_prefix_tokens: str = "<END",
        strip_tokens: bool = True,
        stream_prefix: bool = False,
        sender: str
    ) -> None:
        """Instantiate EofStreamingStdOutCallbackHandler.

        Args:
            answer_prefix_tokens: Token sequence that prefixes the anwer.
                Default is ["Final", "Answer", ":"]
            end_of_file_token: Token that signals end of file.
                Default is "END"
            strip_tokens: Ignore white spaces and new lines when comparing
                answer_prefix_tokens to last tokens? (to determine if answer has been
                reached)
            stream_prefix: Should answer prefix itself also be streamed?
        """
        super().__init__(answer_prefix_tokens=answer_prefix_tokens, strip_tokens=strip_tokens, stream_prefix=stream_prefix)
        self.end_prefix_tokens = end_prefix_tokens
        self.end_reached = False
        self.sender = sender

    def append_to_last_tokens(self, token: str) -> None:
        self.last_tokens.append(token)
        self.last_tokens_stripped.append(token.strip())
        if len(self.last_tokens) > 5:
            self.last_tokens.pop(0)
            self.last_tokens_stripped.pop(0)

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.answer_reached = False
        self.end_reached = False

    def check_if_answer_reached(self) -> bool:
        if self.strip_tokens:
            return ''.join(self.last_tokens_stripped) in self.answer_prefix_tokens_stripped
        else:
            unfied_last_tokens = ''.join(self.last_tokens)
            try:
                unfied_last_tokens.index(self.answer_prefix_tokens)
                return True
            except:
                return False
            
    def check_if_end_reached(self) -> bool:
        if self.strip_tokens:
            return ''.join(self.last_tokens_stripped) in self.answer_prefix_tokens_stripped
        else:
            unfied_last_tokens = ''.join(self.last_tokens)
            try:
                unfied_last_tokens.index(self.end_prefix_tokens)
                self.sender[1] = True
                return True
            except:
                return False

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        # Remember the last n tokens, where n = len(answer_prefix_tokens)
        self.append_to_last_tokens(token)
        
        # Check if the last n tokens match the answer_prefix_tokens list ...
        if not self.answer_reached and self.check_if_answer_reached():
            self.answer_reached = True
            if self.stream_prefix:
                for t in self.last_tokens:
                    sys.stdout.write(t)
                sys.stdout.flush()
            return
        
        if not self.end_reached and self.check_if_end_reached():
            self.end_reached = True

        if self.end_reached:
            pass
        elif self.answer_reached:
            if self.last_tokens[-2] == ":":
                pass
            else:
                self.sender[0] += self.last_tokens[-2]

class Agent:
    def __init__(self, tools, template_path='./templates/agent_prompt_templete.txt', verbose=False):
        config = configparser.ConfigParser()
        config.read('../secrets.ini')
        openai_api_key = config['OPENAI']['OPENAI_API_KEY']
        serper_api_key = config['SERPER']['SERPER_API_KEY']
        serp_api_key = config['SERPAPI']['SERPAPI_API_KEY']
        kakao_api_key = config['KAKAO_MAP']['KAKAO_API_KEY']
        
        template_path = './templates/agent_prompt_templete.txt'
        with open(template_path, 'r') as f:
            template = f.read()

        prompt = CustomPromptTemplate(
            template=template,
            tools=tools,
            input_variables=["input", "intermediate_steps", "conversation_history", "stage_number"]
        )
        
        output_parser = CustomOutputParser()

        llm_chain = LLMChain(llm=ChatOpenAI(model='gpt-4', temperature=0.5, streaming=True), prompt=prompt, verbose=verbose,)

        tool_names = [tool.name for tool in tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain, 
            output_parser=output_parser,
            stop=["\nObservation:"], 
            allowed_tools=tool_names
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)

    async def arun(self, sender, *args, **kwargs):
        resp = await self.agent_executor.arun(kwargs, callbacks=[CustomStreamingStdOutCallbackHandler(answer_prefix_tokens='이우선:', end_prefix_tokens='<END', strip_tokens=False, sender=sender)])
        return resp
    
    def run(self, sender, *args, **kwargs):
        resp = self.agent_executor.run(kwargs)
        return resp