## 우리가 무얼 하는가
와인에 대한 지식이 없는 사람도 와인을 쉽게 즐길 수 있도록 개인의 와인 구매와 성향파악을 돕고, 추천받은 와인을 구할 수 있는 가장 가까운 매장의 정보 제공. 

## 우리 제품의 기능
1. 와인 추천: 사용자의 성향을 파악하고, 적절한 와인을 추천한다. 한국내에서 구매할 수 있는 와인을 추천하고, 구매 링크 제공을 목표로 한다.
2. 와인 Q&A: 사용자가 질문을 하면 그에 적절한 정보를 제공한다.
3. 와인바 추천: 사용자의 상황, 위치 등에 따라서 적절한 와인바를 추천한다.

빠른 제품 개발을 위해 [Langchain](https://github.com/hwchase17/langchain)을 사용한다. Langchain은 언어 모델을 사용하는 제품 개발에 도움을 주는 프레임워크이다.

### 아래는 Langchain 학습 자료
먼저 deeplearning AI 강의를 듣고, documentation의 Quickstart와 Modules부분을 천천히 따라가 보는 것을 추천한다.
- documentation: [documenatation 링크](https://python.langchain.com/docs/get_started/quickstart) <br>
- Deeplearning AI 강의(2시간): [강의링크](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)
- 우리모델과 비슷한 구조로 구현된 자료(Sales GPT): [Sales GPT](https://python.langchain.com/docs/use_cases/agents/sales_agent_with_context)

## 우리 모델 구조
<img src="./assets/architecture.png" width="500" height="500">

### 구성 요소
- Assistant: 대화 기록을 보고, 다음에 Agent가 취해야할 적절한 행동을 출력한다.
- Agent: 대화 기록과 Assistant에서 출력한 행동에 따라 도구를 활용해 유저에게 전달할 적절한 응답을 생성한다.
- Tools: 외부 데이터를 활용하기 위한 도구
- Database: 와인, 와인바 등의 정보가 담겨있는 데이터베이스
- Web: 말그대로 웹(구글)

## requirements

- pip install langchain
- pip install openai
- pip install ipykernel
- pip install pandas
- pip install chromadb
- pip install tiktoken
- pip install lark
- pip install "langchain[docarray]"
- pip install google-search-results
- pip install gradio

### 가장 먼저 [quickstart 노트북](quickstart.ipynb)을 통해 전체 모듈을 파악해보세요! 

## TODO
### Assistant
- [Assistant 노트북](./assistant.ipynb)
- 목표: Assistant evaluation set에 대해 95% 이상으로 Agent의 다음 action을 예측

### 대화 데이터셋 제작
- [Dialog dataset 제작 노트북](./generate_dialog.ipynb)
- 목표: 우리 챗봇에서 일어날만한 대화 데이터셋 100개 제작

### Evaluation Set 제작
- [Evaluation Set 제작 노트북](./generate_evaluation.ipynb)
- 목표: Assistant와 Agent evaluation set 제작

### Tools 및 알고리즘 제작
- [Langchain Tools 노트북](./tools.ipynb)

### 와인 및 와인바 database 제작
- [wine database 제작 노트북](./crawl_wine.ipynb)
- [winebar database 제작 노트북](./crawl_winebar.ipynb)
- 목표: 크롤링을 통한 와인 및 와인바 database 제작

### Agent
- [Agent 노트북](./agent.ipynb)

### Gradio를 통한 웹데모 제작
- [Gradio 배포 노트북](./gradio_web.ipynb)
- 목표: 앱 제작 전 언어모델 테스트 및 GPT4로 생성한 데이터셋 사람이 검증 및 수정