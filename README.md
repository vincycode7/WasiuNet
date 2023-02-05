# Project: WasiuNet - AI Trading Transformer Crypto Auto Trader 

## Technical Details

    Revolutionize your crypto trading experience with our custom Transformer-based microservice solution. Our innovative in-house model blends the power of CNN and Transformers, utilizing multi-timeframe trading data for thorough research and analysis. This solution is easily adaptable to various currencies and model sizes, making it a versatile tool for traders.

    What sets our solution apart is the unique approach to data input. By analyzing data in multiple time frames, our solution creates a sequence of data from 1 min, 5 mins, and 15 mins time frames, which is then processed by a FeedForward + CNN layer to output new height, weight, and channel data. This data is then passed through another CNN layer to produce patches that are finally fed into the Transformers for encoding and decoding.

    Experience the cutting-edge technology in crypto trading with our solution. It leverages the latest advancements in NLP, Deep Learning, Machine Learning, AI, Docker, Pytorch, Github, Redis, Postgresql, Python and Transformers to deliver unparalleled results. Check out our solution on GitHub: https://github.com/vincycode7/WasiuNet, and stay ahead in the race of innovation in the crypto world.

    This project delivers a comprehensive machine learning and data engineering solution, from end to end. It begins with input processing and continues through the training pipeline, resulting in a robust tested and validated solution. The final step is to load the solution into a user-friendly frontend, allowing for tracking of safe trading entry points. With customizable notifications, the system can alert users to safe entry points or even execute trades based on user settings. This pipeline ensures an efficient and effective approach to trading and investment.

## Full Project Documentation

### Introduction
    WasiuNet is a full-scale, industry-standard AI trading transformer solution based on a microservice architecture. The goal of this project is to revolutionize the crypto trading experience by using the latest advancements in NLP, Deep Learning, Machine Learning, AI, Docker, PyTorch, Github, Redis, Postgresql, Python, and Transformers.

### Services

    Authentication (Auth) Service: This service handles user authentication and authorization, ensuring that only authorized users have access to the trading platform and that their actions are secured.

    Machine Learning (ML) Service: This service performs machine learning and artificial intelligence tasks. It generates prediction data that is used by the SafePoint Tracker and Auto Trader services.

    Frontend Service: This service provides a user-friendly interface for the trading platform. Users can interact with the platform, monitor their trades, and execute trades.

    SafePoint Tracker Service: This service checks for safe trade points based on the ML service's prediction data. It continuously monitors the prediction data and updates the Auto Trader service when a safe trade point is detected.

    Auto Trader Service: This service executes safe trades and terminates unhealthy trades. It uses the prediction data from the ML service and updates from the SafePoint Tracker service to determine when to execute trades.

    Auto Notify Service: This service is responsible for all notification-based features, including notifications for safe trade points, trade executions, terminations, and login notifications.

### Microservices 
- [**auth** (For signups and signins)](auth)
    - [Auth Tests](auth/tests)

- [**frontend** (For visual display)](frontend)
    - [Frontend Tests](frontend/tests)

- [**ml** (For model training, testing, validation and running predictions)](ml)
    - [ML Tests](ml/tests)

- [**safepoint_tracker** (Acts as a form of tracker to compare previous model output, along side previous predicted safepoints to predict the new safe point.)](safepoint_tracker)
    - [Safepoint Tracker Tests](safepoint_tracker/tests)

- [**auto_trader** (Acts as a form of auto trading to automate safetrades and close unhealther trades.)](auto_trader)
    - [Auto Trader Tests](auto_trader/tests)

- [**auto_notify** (Acts as a form of notification system to handle any form of notification in the entire system, notification ranging from, safe point detection, auto safe trade executed, auto unhealthy trade close, signin notification, reset password notification, forgot password notification, news notification, etc.)](auto_notify)
    - [Auto Notify Tests](auto_notify/tests)


## Relationship between services
    
    The Auth, ML, and Frontend services are all connected to the SafePoint Tracker, Auto Trader, and Auto Notify services. The SafePoint Tracker and Auto Trader services both rely on the ML service's prediction data, while the Auto Notify service receives updates from the SafePoint Tracker and Auto Trader services. All of these services work together to provide a comprehensive and automated trading solution.

## Project-Plan
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

## General System Architecture
- None

## Installation
- Clone Repo using `git clone https://github.com/vincycode7/WasiuNet.git`
- [Setup and Run Frontend](frontend/README.md)
- [Setup and Run ML](ml/README.md)
- [Setup and Run SafePoint_Tracking](safepoint_tracker/README.md)
- [Setup and Run Auth](auth/README.md)
- [Setup and Run Auto_Trader](auto_trader/README.md)
- [Setup and Run Auto_Notifier](auto_notifier/README.md)

## Conclusion

    WasiuNet offers a comprehensive and automated trading solution for cryptocurrency, providing a robust and reliable trading experience. The microservice architecture ensures scalability and ease of maintenance, while the combination of AI and machine learning delivers unparalleled results.

## Note: 

-   The ML part of this project is currently under development on Google Colab. Follow this link to access it [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1Nm_8_5firMCZ3w-A0y-AdrE2g0VBJT4d/view?usp=sharing)

-   `pipenv` package is adviced to be used for development
-   If contributing, remember to add any new package you introduced to the project on your local pc into the requirement file for each microservices by 
    * `cd` to the micro service folder you are interested in
    * activate your environment using `pipenv shell` or the equivalent command depending on the package you are using, 
    * once environment is activated then you run `pipenv run pip freeze > requirements.txt`(or equivalent) to push all the required package from your environment into the `requirement.txt` file.

## Resources:

-   [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473.pdf)

-   [Attention is all you need paper](https://arxiv.org/pdf/1706.03762.pdf)

-   [Are Transformers Effective for Time Series Forecasting](https://arxiv.org/pdf/2205.13504.pdf)

-   [Visual Guide to Transformer Neural Networks - (Episode 1) Position Embeddings](https://www.youtube.com/watch?v=dichIcUZfOw)

-   [Transformers in Vision: From Zero to Hero](https://www.youtube.com/watch?v=J-utjBdLCTo)

-   [Pytorch seq2seq implementation series by Aladdin Persson](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbnM2SXZwZTFfbG1FZkN2RXVsemYySlNJa2kxd3xBQ3Jtc0ttbUoySDNmbGF4V2d6WS0xWTZQOG1SUlBvMzZ1STd6MzhJTWJhM3JOZ0kxU0FCRGlWS2k1VFBQako5TkNHaURySVlSSU1Sa3pOR0wwai1sV1JGcV85UDdpTV9xRGs3SldMdm9reTBTQWVoalZwSFd6dw&q=https%3A%2F%2Fgithub.com%2Faladdinpersson%2FMachine-Learning-Collection&v=U0s0f995w14)

-   [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

-   [PyTorch Paper Replicating - building a vision transformer with PyTorch](https://youtu.be/tjpW_BY8y3g)

-   [Microservices in Python using Flask Framework | Dockerize and Deploy to Kubernetes with Helm](https://www.youtube.com/watch?v=SdTzwYmsgoU&list=PL8klaCXyIuQ4RYLGVJUO_iOkmumkXKjPY&index=2)

-   [Kubernetes Tutorial for Beginners FULL COURSE in 4 Hours](https://www.youtube.com/watch?v=X48VuDVv0do)