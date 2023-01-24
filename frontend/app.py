import os, sys, aiohttp, asyncio
import streamlit as st
from config.config import PREDICT_ENDPOINT
from datetime import datetime
from aiohttp.client_exceptions import ClientConnectorError
# from  data_eng import data_util
# sys.path.append(os.path.abspath(os.path.join(os.path.abspath(''), '..'))) # 


async def get_prediction(session, input_data, auth_token=None):
    # """
    # Get a prediction from the ML service
    # :param input_data: str - the input data for the prediction
    # :param auth_token: str - the auth token to be used for authentication
    # :param session: aiohttp.ClientSession - an aiohttp session object
    # :return: dict - the ML service's response
    # """
    # headers = {'Authorization': auth_token}
    ml_data = {'input': input_data}
    # async with session.post(ml_url, json=ml_data, headers=headers) as ml_response:
    try:
        async with session.post(PREDICT_ENDPOINT, json=ml_data) as ml_response:
            ml_json = await ml_response.json()
    except ClientConnectorError as e:
        return {"error":"Internal Server Error","message":f"ClientConnectorError occured, check if ml service is running. {e}"}, 500
    return ml_json, 200

# @st.cache
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
    input_data = {"asset" : asset, "date_time" : date_time}
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
ml_res, status = asyncio.run(get_ml_prediction_streamlit(input_data=input_data))

if status!=200:
    st.error("An error occurred while connecting to the ML Predict Service Endpoint: " + str(ml_res['error']))
else:
    st.write(ml_res)
    
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
# st.success('This is a success message!', icon="✅")

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