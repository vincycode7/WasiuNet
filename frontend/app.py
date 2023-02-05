import os, sys, aiohttp, asyncio, tracemalloc

import streamlit as st
from config.config import PREDICT_ENDPOINT
from datetime import datetime
from aiohttp.client_exceptions import ClientConnectorError, ContentTypeError
# from  data_eng import data_util

async def get_prediction(session, input_data, auth_token=None):
    # """
    # Get a prediction from the ML service
    # :param input_data: str - the input data for the prediction
    # :param auth_token: str - the auth token to be used for authentication
    # :param session: aiohttp.ClientSession - an aiohttp session object
    # :return: dict - the ML service's response
    # """
    # headers = {'Authorization': auth_token}
    ml_data = input_data
    # async with session.post(ml_url, json=ml_data, headers=headers) as ml_response:
    try:
        async with session.post(PREDICT_ENDPOINT, json=ml_data) as ml_response:
            ml_json = await ml_response.json()
    except ClientConnectorError as e:
        return {"error":"Internal Server Error","message":f"ClientConnectorError occured, check if ml service is running. {e}"}, 500
    except ContentTypeError as e:
        return {"error":"Internal Server Error", "message":f"ContentTypeError occured, check if ml service is sending a valid response. {e}"}, 500
    return ml_json, 200

def get_ml_prediction_streamlit(input_data):
    async def _get_ml_prediction_streamlit():
        async with aiohttp.ClientSession() as session:
            return await get_prediction(session, input_data)
    return _get_ml_prediction_streamlit()

def get_data_streamlit(canvas):
    # Get inputs
    col1, col2, col3 = canvas.columns([7,5,4])
    col2.markdown("### WasiuNet")
    col = canvas.columns([1,5,5,5,2,1])
    asset = col[1].selectbox("select asset",["btc-usd", "eth-usd"])
    trade_date = col[2].date_input("start safe entry search - date")
    trade_time = col[3].time_input("start safe entry search - time")
    date_time = datetime.combine(trade_date, trade_time).strftime("%Y-%m-%d-%H-%M-%S")
    col = col[4].select_slider("Auto safe trade",["Off","On"])
    input_data = {"asset" : asset, "pred_datetime" : date_time}
    return input_data

st.set_page_config(
    page_title="WasiuNet",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

input_data = get_data_streamlit(canvas=st)
st.write("out",input_data)

tracemalloc.start()
ml_res, status = asyncio.run(get_ml_prediction_streamlit(input_data=input_data))
tracemalloc.stop()

if status!=200:
    st.error("An error occurred while connecting to the ML Predict Service Endpoint: " + str(ml_res['error']))
    print(f"Error message: {str(ml_res['message'])}")
else:
    st.success(f'{ml_res}!', icon="✅")
    
    
# import streamlit as st
# import requests
# import json
# import time

# async def fetch_prediction(data):
#     url = "http://localhost:5000/predict"
#     headers = {'content-type': 'application/json'}
#     response = requests.post(url, data=json.dumps(data), headers=headers)
#     return response.json()

# async def fetch_result(prediction_key):
#     url = f"http://localhost:5000/result/{prediction_key}"
#     response = requests.get(url)
#     return response.json()

# async def main():
#     st.title("Crypto Trading Predictions")
#     data = {
#         "feature1": st.number_input("Enter feature 1 value"),
#         "feature2": st.number_input("Enter feature 2 value"),
#         "feature3": st.number_input("Enter feature 3 value"),
#         "feature4": st.number_input("Enter feature 4 value")
#     }
#     if st.button("Predict"):
#         prediction = await fetch_prediction(data)
#         prediction_key = prediction["prediction_key"]
#         if "prediction" in prediction:
#             st.write("Prediction: ", prediction["prediction"])
#         else:
#             result = None
#             while not result:
#                 result = await fetch_result(prediction_key)
#                 time.sleep(1)
#             st.write("Prediction: ", result["prediction"])

# if __name__ == "__main__":
#     asyncio.run(main())

# st.write("ML_BASE_URL"+":"+"ML_PORT" + f"--> {ML_BASE_URL}:{ML_PORT}")

# async def authenticate_user(session, username, password):
#     """
#     Authenticate a user with the auth service
#     :param session: aiohttp.ClientSession - an aiohttp session object
#     :param username: str - the user's username
#     :param password: str - the user's password
#     :return: dict - the auth service's response
#     """
#     auth_url = 'http://auth-service:5000/authenticate'
#     auth_data = {'username': username, 'password': password}
#     async with session.post(auth_url, json=auth_data) as auth_response:
#         auth_json = await auth_response.json()
#     return auth_json

# Send input to ml microsevice with inputs
# user_token, asset, date, time,run_prediction_flag endpoint and either get just data or data and prediction 



# st.write(data_util.data_util.convert_date_from_backend_format(trade_date)+"-"+trade_time.replace(":","-"))
# st.error('This is an error', icon="🚨")

# if not st.checkbox("proceed"):
#     st.stop()

# with st.form("my_form"):
#    st.write("Inside the form")
#    slider_val = st.slider("Form slider")
#    checkbox_val = st.checkbox("Form checkbox")

#    # Every form must have a submit button.
#    submitted = st.form_submit_button("Submit")
#    if submitted:
#        st.write("slider", slider_val, "checkbox", checkbox_val)

# st.write("Outside the form")

# def get_user_name():
#     return 'John'

# with st.echo():
#     # Everything inside this block will be both printed to the screen
#     # and executed.

#     def get_punctuation():
#         return '!!!'

#     greeting = "Hi there, "
#     value = get_user_name()
#     punctuation = get_punctuation()

#     st.write(greeting, value, punctuation)