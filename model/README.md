# ChatWine Model

![model](/assets/architecture.png)

## Assistant
대화 기록을 보고, 다음에 Agent가 취해야할 적절한 행동을 추론합니다.

Agent의 행동은 시작하기, 분석, 가격대 확인, 와인 추천, 판매, 위치 제안, 대화 마무리하기, 질문 답변, 기타 상황 의 9가지 상황으로 구성됩니다.

Assistant python 모듈 코드는 [여기](https://github.com/audreyaiai/chatwine-dev/blob/main/model/assistant.py)에서 확인할 수 있고, 튜토리얼 노트북은 [여기](https://github.com/audreyaiai/chatwine-dev/blob/main/tutorials/assistant.ipynb) 에서 확인할 수 있습니다.

## Agent
대화 기록과 Assistant에서 추론한 행동에 따라 도구를 활용해 유저에게 전달할 적절한 응답을 생성하고, 그에 대한 사용자의 응답을 예측하여 추천합니다.

Agent python 모듈 코드는 [여기](https://github.com/audreyaiai/chatwine-dev/blob/main/model/agent.py) 에서 확인할 수 있고, 튜토리얼 노트북은 [여기](https://github.com/audreyaiai/chatwine-dev/blob/main/tutorials/agent.ipynb) 에서 확인할 수 있습니다.


## Tools
외부 데이터를 활용하기 위한 도구입니다.
아래의 도구들에 접근할 수 있습니다.

### Database
와인, 와인바 등의 정보가 담겨있는 데이터베이스입니다.

와인 데이터에 대한 설명은 [여기](https://github.com/audreyaiai/chatwine-dev/blob/main/data/README.md)에서 확인할 수 있고, 실제 데이터 파일은 [여기](https://github.com/audreyaiai/chatwine-dev/blob/main/data/unified_wine_data.json) 에서 확인할 수 있습니다.


### Web
웹(구글)에서 필요한 정보를 검색하여 제공합니다.

Tools python 모듈 코드는 [여기](https://github.com/audreyaiai/chatwine-dev/blob/main/model/tools.py) 에서 확인할 수 있고, 튜토리얼 노트북은 [여기](https://github.com/audreyaiai/chatwine-dev/blob/main/tutorials/tools.ipynb) 에서 확인할 수 있습니다.