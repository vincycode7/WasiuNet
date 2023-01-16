# WasiuNet
A time series future asset price estimator, using multiple timeframe datapoint for future price estimation.

Note: Project currently in development in google colab, Follow link below to access, Also note that project on github is still in boilerplate stage, currently structuring folders to take into account microservice deployment using kubernetes, CI/CD using git actions to build and push docker-containers to dockerhub for kubernetes to pull and structuring the test for each micro service.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1Nm_8_5firMCZ3w-A0y-AdrE2g0VBJT4d/view?usp=sharing)

1. Microservices 
    - **auth** (For signups and signins)
        - [Auth Tests](auth/tests)
    - **frontend** (For visual display)
        - [Frontend Tests](frontend/tests)
    - **ml** (For model training, testing, validation and running predictions)
        - [ML Tests](ml/tests)
    - **safepoint_tracker** (Acts as a form of tracker to compare previous model output, along side previous predicted safepoints to predict the new safe point.) 
        - [Safepoint Tracker Tests](safepoint_tracker/tests)
    
2. Project-Plan
    - **Init Project [done]**
    - **Include files from research environment (google colab) [inprogress(continous-development)]**
    - **Init Dummy Project Boiler plate to handle, kubernetes deploymeny, github-gitaction, microservices, testing [done]**
    - **Build out frontend using steamlit [inprogress]**
    - **Implement test cases for frontend [not-started]**
    - **Build out ml using research code and a flask webapp to recieve and send out data [not-started]**
    - **Implement test cases for ml [not-started]**
    - **Build out safepoint_tracker to track ml output fed from frontend [not-started]**
    - **Implement test cases for safepoint_tracker [not-started]**
    - **Introduce database for all microservices [not-started]**
    - **mock database testing for all microservices [not-started]**
    - **Build out Flask app auth microservice to signup, login and give tokens, this will help restrict access to all the other services [not-started]**
    - **mock auth testing for all auth microservices [not-started]**
    - **deploy [not-started]**

Resources:

(1) [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473.pdf)

(2) [Attention is all you need paper](https://arxiv.org/pdf/1706.03762.pdf)

(2) [Are Transformers Effective for Time Series Forecasting](https://arxiv.org/pdf/2205.13504.pdf)

(3) [Visual Guide to Transformer Neural Networks - (Episode 1) Position Embeddings](https://www.youtube.com/watch?v=dichIcUZfOw)

(4) [Transformers in Vision: From Zero to Hero](https://www.youtube.com/watch?v=J-utjBdLCTo)

(5) [Pytorch seq2seq implementation series by Aladdin Persson](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbnM2SXZwZTFfbG1FZkN2RXVsemYySlNJa2kxd3xBQ3Jtc0ttbUoySDNmbGF4V2d6WS0xWTZQOG1SUlBvMzZ1STd6MzhJTWJhM3JOZ0kxU0FCRGlWS2k1VFBQako5TkNHaURySVlSSU1Sa3pOR0wwai1sV1JGcV85UDdpTV9xRGs3SldMdm9reTBTQWVoalZwSFd6dw&q=https%3A%2F%2Fgithub.com%2Faladdinpersson%2FMachine-Learning-Collection&v=U0s0f995w14)

(6) [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

(7) [PyTorch Paper Replicating - building a vision transformer with PyTorch](https://youtu.be/tjpW_BY8y3g)