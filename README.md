# WasiuNet
Revolutionize your crypto trading experience with our custom Transformer-based microservice solution. Our innovative in-house model blends the power of CNN and Transformers, utilizing multi-timeframe trading data for thorough research and analysis. This solution is easily adaptable to various currencies and model sizes, making it a versatile tool for traders.

What sets our solution apart is the unique approach to data input. By analyzing data in multiple time frames, our solution creates a sequence of data from 1 min, 5 mins, and 15 mins time frames, which is then processed by a FeedForward + CNN layer to output new height, weight, and channel data. This data is then passed through another CNN layer to produce patches that are finally fed into the Transformers for encoding and decoding.

Experience the cutting-edge technology in crypto trading with our solution. It leverages the latest advancements in NLP, Deep Learning, Machine Learning, AI, Docker, Pytorch, Github, Redis, Postgresql, Python and Transformers to deliver unparalleled results. Check out our solution on GitHub: https://github.com/vincycode7/WasiuNet, and stay ahead in the race of innovation in the crypto world.

This project delivers a comprehensive machine learning and data engineering solution, from end to end. It begins with input processing and continues through the training pipeline, resulting in a robust tested and validated solution. The final step is to load the solution into a user-friendly frontend, allowing for tracking of safe trading entry points. With customizable notifications, the system can alert users to safe entry points or even execute trades based on user settings. This pipeline ensures an efficient and effective approach to trading and investment.

Note: Project currently in development in google colab, Follow link below to access, Also note that project on github is still in boilerplate stage, currently structuring folders to take into account microservice deployment using kubernetes, CI/CD using git actions to build and push docker-containers to dockerhub for kubernetes to pull and structuring the test for each micro service.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1Nm_8_5firMCZ3w-A0y-AdrE2g0VBJT4d/view?usp=sharing)

1. Microservices 
    - [**auth** (For signups and signins)](auth)
        - [Auth Tests](auth/tests)
    - [**frontend** (For visual display)](frontend)
        - [Frontend Tests](frontend/tests)
    - [**ml** (For model training, testing, validation and running predictions)](ml)
        - [ML Tests](ml/tests)
    - [**safepoint_tracker** (Acts as a form of tracker to compare previous model output, along side previous predicted safepoints to predict the new safe point.)](safepoint_tracker)
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

3. General System Architecture
    - None

4. Installation
    - Clone Repo using `git clone https://github.com/vincycode7/WasiuNet.git`
    - [Setup and Run Frontend](frontend/README.md)
    - [Setup and Run ML](ml/README.md)
    - [Setup and Run SafePoint_Tracking](safepoint_tracker/README.md)
    - [Setup and Run Auth](auth/README.md)

5. Notes
    - `pipenv` package is adviced to be used for development
    - If contributing, remember to add any new package you introduced to the project on your local pc into the requirement file for each microservices by 
        - cd to the micro service folder you are interested in
        - activate your environment using `pipenv shell` or the equivalent command depending on the package you are using, 
        - once environment is activated then you run `pipenv run pip freeze > requirements.txt`(or equivalent) to push all the required package from your environment into the `requirement.txt` file.

6. Resources:

    (1) [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473.pdf)

    (2) [Attention is all you need paper](https://arxiv.org/pdf/1706.03762.pdf)

    (3) [Are Transformers Effective for Time Series Forecasting](https://arxiv.org/pdf/2205.13504.pdf)

    (4) [Visual Guide to Transformer Neural Networks - (Episode 1) Position Embeddings](https://www.youtube.com/watch?v=dichIcUZfOw)

    (5) [Transformers in Vision: From Zero to Hero](https://www.youtube.com/watch?v=J-utjBdLCTo)

    (6) [Pytorch seq2seq implementation series by Aladdin Persson](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbnM2SXZwZTFfbG1FZkN2RXVsemYySlNJa2kxd3xBQ3Jtc0ttbUoySDNmbGF4V2d6WS0xWTZQOG1SUlBvMzZ1STd6MzhJTWJhM3JOZ0kxU0FCRGlWS2k1VFBQako5TkNHaURySVlSSU1Sa3pOR0wwai1sV1JGcV85UDdpTV9xRGs3SldMdm9reTBTQWVoalZwSFd6dw&q=https%3A%2F%2Fgithub.com%2Faladdinpersson%2FMachine-Learning-Collection&v=U0s0f995w14)

    (7) [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

    (8) [PyTorch Paper Replicating - building a vision transformer with PyTorch](https://youtu.be/tjpW_BY8y3g)

    (9) [Microservices in Python using Flask Framework | Dockerize and Deploy to Kubernetes with Helm](https://www.youtube.com/watch?v=SdTzwYmsgoU&list=PL8klaCXyIuQ4RYLGVJUO_iOkmumkXKjPY&index=2)

    (10) [Kubernetes Tutorial for Beginners FULL COURSE in 4 Hours](https://www.youtube.com/watch?v=X48VuDVv0do)