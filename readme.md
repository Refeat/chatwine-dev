<!-- HEADER START -->
<p align="center"><a href="#">
    <img width="100%" height="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:B993D6,100:8CA6DB&height=220&section=header&fontSize=40&fontColor=ffffff&animation=fadeIn&fontAlignY=40&text=ChatWine" alt="header" />
</a></p>

<!-- HEADER END -->

# ❓ ChatWine은 어떤 서비스인가요?
와인에 대한 지식이 없는 사람도 와인을 쉽게 즐길 수 있도록 개인의 와인 구매와 성향파악을 돕고, 추천받은 와인을 구할 수 있는 가장 가까운 매장의 정보를 제공합니다. 

## 🍷 ChatWine 의 기능
1. 와인 추천: 사용자의 성향을 파악하고, 적절한 와인을 추천합니다. 한국내에서 구매할 수 있는 와인을 추천하고, 구매 링크나 바로 구매하는 기능을 제공합니다.
2. 와인 Q&A: 사용자가 와인에 대한 질문을 하면 그에 적절한 정보를 제공합니다.
3. 와인바 추천: 사용자의 상황, 위치 등에 따라서 적절한 와인바를 추천합니다.

<br><br>

# 👋 TEAM - audrey.ai
2년간 인공지능 동아리에서 호흡을 맞춰 왔고, 높은 실행력을 바탕으로 유수 AI 스타트업과 산학협력, AI 대회 및 해커톤에서 수상했던 경험이 있는 연세대학교에서 가장 우수한 AI팀

|이름|역할|소개|연락처|
|:---:|:---:|:---|:---|
|이상민|CoFounder<br>대표<br>영업<br>개발|- 연세대학교 물리학 전공<br>- 연세대학교 인공지능 동아리 회장단<br>- 로민 산학협력 OCR 모델<br>- 포자랩스 산학협력 미디 수정 모델<br>-  연세대학교 인공지능 학회 주체 인공지능 경진대회 1등<br>- 서울대학교 SNU IGNITE Build-up (인공지능 팀으로 참여)|✉️ sangmin23@yonsei.ac.kr|
|최성범|CoFounder<br>부대표<br>제품 개발|- 연세대학교 천문우주학 전공<br>- (전) 뉴빌리티 선임 연구원<br>프로젝트 경험<br>- 연세대학교 인공지능 경진대회 3등 (팀장)<br>- 현대모비스 해커톤 2등 (차량 내 음성인식)<br>- Hands-Free 실시간 번역 모델 개발<br>- AI CV 기업 알체라 Blur face detection 모델 개발|✉️ choisb3631@gmail.com|
|박찬혁|팀장<br>소프트웨어 개발|- 연세대학교 전기 전자 공학 전공<br>- (전) 리빌더AI 개발자<br>- 연세대학교 인공지능 학회 주최 인공지능 경진대회 1등<br>-  Diffusion 기반 스케치에서 이미지로 바꿔주는 모델/웹 개발|✉️ devch1013@gmail.com|
|박민수|AI 개발|- 연세대학교 컴퓨터과학 전공<br>- 연세대학교 인공지능대학원 연구실 인턴<br>- 연세대학교 인공지능 경진대회 3등<br>프로젝트 경험<br>- Diffusion 기반 목소리 변환 노래 생성 모델 개발<br>- AI 스타트업 포자랩스 산학협력<br>(Transformer 기반 음악 생성 모델 개발)<br>- 현대 모비스 해커톤 2등<br>(차량 내 음성인식 모델 개발)<br>- ArcFace 기반 사진분류 앱 개발|✉️ 0601p@naver.com|

<br><br>

# 🔥 Quick Start
## 1, Cloning repository
```
git clone https://github.com/audreyaiai/chatwine-dev.git
cd chatwine
```

## 2, Installing Dependencies
python version 3.10.11 에 설치하는 것을 추천합니다.

### conda environment 생성(필요한 경우만)
```
conda create -n chatwine python=3.10.11
conda activate chatwine
```

### 필요한 모듈 설치
```
pip install -r requirements.txt
```

## 3, Run Web Demo
```
gradio app.py
```

<br><br>

# 🤖 MODEL
![model](/assets/architecture.png)

## 구성 요소
- Assistant : 대화 기록을 보고, 다음에 Agent가 취해야할 적절한 행동을 추론합니다.
- Agent : 대화 기록과 Assistant에서 추론한 행동에 따라 도구를 활용해 유저에게 전달할 적절한 응답을 생성하고, 그에 대한 사용자의 응답을 예측하여 추천합니다.
- Tools : 외부 데이터를 활용하기 위한 도구입니다. 데이터베이스에서 정보를 가져오거나 웹에서 정보를 검색합니다.
- Database : 와인, 와인바 등의 정보가 담겨있는 데이터베이스입니다.
- Web : 웹(구글)에서 필요한 정보를 검색하여 제공합니다.

모델에 대한 자세한 정보는 [여기]()에서 확인할 수 있습니다.

<br><br>

# 📋 Data
[와인나라](https://www.winenara.com/), [와인앤모어](https://www.wineandmore.co.kr/), [망고플레이트](https://www.mangoplate.com/), [VIVINO](https://www.vivino.com/) 에서 와인 정보들을 크롤링 해 하나의 와인 데이터셋으로 통합했습니다.

데이터에 대한 자세한 정보는 [여기]()에서 확인할 수 있습니다.