# Define the tools below
# WineDatabaseTool: A tool that searches for wines in the wine database.
# WineBarDatabaseTool: A tool that searches wine bar in seoul.
# SearchTool: A tool that searches for the desired information in the search engine.
# KakaoMapTool: A tool that searches for the destrict information in the KakaoMap.

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
from langchain.callbacks.manager import Callbacks, CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
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
from langchain.tools import BaseTool

class CustomSelfQueryRetriever(SelfQueryRetriever):
    """
    A retriever that uses the LLMChain to generate a query from a human message. 
    It then uses the vectorstore to retrieve documents relevant to that query.    
    """
    async def aget_relevant_documents(self, query: str, callbacks: Callbacks = None) -> List[Document]:
        inputs = self.llm_chain.prep_inputs({"query": query})
        structured_query = cast(
            StructuredQuery,
            self.llm_chain.predict_and_parse(callbacks=callbacks, **inputs),
        )
        if self.verbose:
            print(structured_query)
        new_query, new_kwargs = self.structured_query_translator.visit_structured_query(
            structured_query
        )
        if structured_query.limit is not None:
            new_kwargs["k"] = structured_query.limit

        if self.use_original_query:
            new_query = query

        search_kwargs = {**self.search_kwargs, **new_kwargs}
        docs = self.vectorstore.search(new_query, self.search_type, **search_kwargs)
        return docs


class WineDatabaseTool(BaseTool):
    name = "wine_database"
    description = """
Database about the wines in wine store.
You can search wines with the following attributes:
- price: The price range of the wine. You have to specify greater than and less than.
- rating: 1-5 rating float for the wine. You have to specify greater than and less than.
- wine_type: The type of wine. It can be '레드', '로제', '스파클링', '화이트', '디저트', '주정강화'
- name: The name of wine.
- pairing: The food pairing of wine.
The form of Action Input must be 'key1: value1, key2: value2, ...'. For example, to search for wines with a rating of less than 3 points, a price range of 50000원 or more, and a meat pairing, enter 'rating: gt 0 lt 3, price: gt 50000, pairing: 고기'.
--------------------------------------------------
You can get the following attributes:
- url: Wine purchase site URL.
- vivino_link: Vivino link of wine.
- flavor_description
- site_name: Wine purchase site name.
- name: The name of wine in korean.
- en_name: The name of wine in english.
- price: The price of wine in 원.
- rating: 1-5 vivino rating.
- wine_type: The type of wine.
- pairing: The food pairing of wine.
- pickup_location: Offline stores where you can purchase wine
- img_url
- country
- body
- tannin
- sweetness
- acidity
- alcohol
- grape
The form of Desired Outcome must be 'key1, key2, ...'. For example to get the name and price of wine, enter 'name, price'.
"""
    wine_vectorstore: Any
    wine_retriever: Any

    def __init__(self, data_path:str='./data/unified_wine_data.json', 
                page_content_columns:List=['name', 'pairing'], 
                float_columns:List=['rating', 'price', 'body', 'sweetness', 'alcohol', 'acidity', 'tannin'], 
                drop_columns:List=[],
                verbose:bool=True,
                **data):
        super().__init__(**data)
        embeddings = OpenAIEmbeddings()
        try:
            # raise NotImplementedError()
            self.wine_vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
            print("WineDatabaseTool: Loading vectorstore from disk")
        except:
            ### Load wine database json
            df = pd.read_json(data_path, encoding='utf-8', lines=True)

            # page_content는 encoding되어 similarity를 계산할 때 사용된다.
            df['page_content'] = ''

            for column in page_content_columns:
                if column != 'page_content':
                    df['page_content'] += column + ':' + df[column].astype(str) + ','

            # 다음 column은 필터링을 위해 float으로 변환한다.
            for column in float_columns:
                df[column] = df[column].str.replace(',', '')
                df[column] = df[column].apply(lambda x: float(x) if x != '' else -1)


            # 필요없는 column은 삭제한다.
            df = df.drop(columns=drop_columns)

            loader = DataFrameLoader(data_frame=df, page_content_column='page_content')
            docs = loader.load()
            embeddings = OpenAIEmbeddings()

            self.wine_vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
            self.wine_vectorstore.persist()

        document_content_description = "A database of wines. 'name' and 'pairing' must be included in the query, and 'Body', 'Tannin', 'Sweetness', 'Alcohol', 'Price', 'Rating', 'Wine_Type', and 'Country' can be included in the filter. query and filter must be form of 'key: value'. For example, query: 'name: 돔페리뇽, pairing:육류'."
        llm = OpenAI(temperature=0)

        # 아래는 wine database1에 metadata_field Attribute이다. 아래를 기준으로 필터링을 진행하게 된다.
        metadata_field_info = [
                AttributeInfo(
                    name="body",
                    description="1-5 rating for the body of wine",
                    type="int",
                ),
                AttributeInfo(
                    name="tannin",
                    description="1-5 rating for the tannin of wine",
                    type="int",
                ),
                AttributeInfo(
                    name="sweetness",
                    description="1-5 rating for the sweetness of wine",
                    type="int",
                ),
                AttributeInfo(
                    name="alcohol",
                    description="1-5 rating for the alcohol of wine",
                    type="int",
                ),
                AttributeInfo(
                    name="price",
                    description="The price of the wine",
                    type="int",
                ),
                AttributeInfo(
                    name="rating", 
                    description="1-5 rating for the wine", 
                    type="float"
                ),
                AttributeInfo(
                    name="wine_type", 
                    description="The type of wine. It can be '레드', '로제', '스파클링', '화이트', '디저트', '주정강화'", 
                    type="string"
                ),
                AttributeInfo(
                    name="country", 
                    description="The country of wine. It can be '기타 신대륙', '기타구대륙', '뉴질랜드', '독일', '미국', '스페인', '아르헨티나', '이탈리아', '칠레', '포루투칼', '프랑스', '호주'", 
                    type="float"
                ),
            ]

        self.wine_retriever = CustomSelfQueryRetriever.from_llm(
            llm, self.wine_vectorstore, document_content_description, metadata_field_info, verbose=verbose
        )  # Added missing closing parenthesis        

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return self.wine_retriever.get_relevant_documents(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return await self.wine_retriever.aget_relevant_documents(query)

class WineBarDatabaseTool(BaseTool):
    name = "wine_bar_database"
    description = """
Database about the winebars in Seoul. It should be the first thing you use when looking for information about a wine bar.

- query: The query of winebar. You can search wines with review data like mood or something.
- name: The name of winebar.
- price: The average price point of a wine bar.
- rating: 1-5 rating float for the wine bar. 
- district: The district of wine bar. Input district must be korean. For example, if you want to search for wines in Gangnam, enter 'district: 강남구'
The form of Action Input must be 'key1: value1, key2: value2, ...'. 
--------------------------------------------------
You can get the following attributes:
- name: The name of winebar.
- url: Wine purchase site URL.
- rating: 1-5 망고플레이트(맛집검색 앱) rating.
- summary: Summarized information about wine bars
- address
- phone
- parking
- opening_hours
- menu
- holidays
- img_url
The form of Desired Outcome must be 'key1, key2, ...'. For example to get the name and price of wine, enter 'name, price'.
"""
    wine_bar_vectorstore: Any
    wine_bar_retriever: Any

    def __init__(self, data_path:str='./data/wine_bar_data.json', 
                page_content_columns:List=['summary'],
                float_columns:List=[],
                drop_columns:List=['review'],
                verbose:bool=True,
                **data):
        super().__init__(**data)        
        embeddings = OpenAIEmbeddings()
        try:
            # raise NotImplementedError()
            self.wine_bar_vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
            print("WineBarDatabaseTool: Loading vectorstore from disk")
        except:
            ### Load wine database json
            df = pd.read_json(data_path, encoding='utf-8', lines=True)

            # page_content는 encoding되어 similarity를 계산할 때 사용된다.
            df['page_content'] = ''
            for column in page_content_columns:
                if column != 'page_content':
                    df['page_content'] += df[column].astype(str) + ','

            # 다음 column은 필터링을 위해 float으로 변환한다.
            for idx in df.index:
                for column in float_columns:
                    df[column][idx] = float(df[column][idx]) if df[column][idx] != '' else -1

            # 필요없는 column은 삭제한다.
            df = df.drop(columns=drop_columns)

            loader =DataFrameLoader(data_frame=df, page_content_column='page_content')
            docs = loader.load()
            
            self.wine_bar_vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
            self.wine_bar_vectorstore.persist()
        document_content_description = "Database of a winebar"
        llm = OpenAI(temperature=0)

        metadata_field_info = [
                AttributeInfo(
                    name="name",
                    description="The name of the wine bar",
                    type="str",
                ),
                AttributeInfo(
                    name="rating", 
                    description="1-5 rating for the wine bar", 
                    type="float"
                ),
                AttributeInfo(
                    name="district",
                    description="The district of the wine bar.",
                    type="str",
                ),
            ]
        
        self.wine_bar_retriever = CustomSelfQueryRetriever.from_llm(
            llm, self.wine_bar_vectorstore, document_content_description, metadata_field_info=metadata_field_info, verbose=verbose
        )

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return self.wine_bar_retriever.get_relevant_documents(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return await self.wine_bar_retriever.aget_relevant_documents(query)

class SearchTool(BaseTool):
    name = "search"
    description = "Useful for when you need to ask with search."
    search: Any

    def __init__(self, **data):
        super().__init__(**data)
        self.search = SerpAPIWrapper()

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return self.search.run(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise await self.search.arun(query)

class KakaoMapTool(BaseTool):
    name: str = "map"
    description: str = "The tool used to draw a district for a region. When looking for wine bars, you can use this before applying filters based on location. The query must be in Korean. You can get the following attribute: district."
    url: str = 'https://dapi.kakao.com/v2/local/search/keyword.json' 
    kakao_api_key: str = os.getenv('KAKAO_API_KEY')
    headers : dict = {"Authorization": f"KakaoAK {kakao_api_key}"}
                
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ):
        params = {'query': query,'page': 1}
        places = requests.get(self.url, params=params, headers=self.headers).json()
        address = places['documents'][0]['address_name']
        if not address.split()[0].startswith('서울'):
            return {'district': 'not in seoul'}
        else:
            return {'district': address.split()[1]}
        
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ):
        async with aiohttp.ClientSession() as session:
            params = {'query': query,'page': 1}
            async with session.get(self.url, params=params, headers=self.headers) as response:
                places = await response.json()
                address = places['documents'][0]['address_name']
                if not address.split()[0].startswith('서울'):
                    return {'district': 'not in seoul'}
                else:
                    return {'district': address.split()[1]}

if __name__ == "__main__":
    tools = [
        WineDatabaseTool(),
        WineBarDatabaseTool(),
        SearchTool(),
        KakaoMapTool(),
    ]