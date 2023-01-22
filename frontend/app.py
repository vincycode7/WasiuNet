import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(''), '..'))) # 
import streamlit as st
# from  data_eng import data_util

# sidebar = st.s
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

# Get inputs
col1, col2, col3 = st.columns([7,5,4])
col2.markdown("### WasiuNet")
col = st.columns([1,5,5,5,2,1])
asset1 = col[1].selectbox("select asset",["btc-usd", "eth-usd"])
trade_date = col[2].date_input("start safe entry search - date")
trade_time = col[3].time_input("start safe entry search - time")
col = col[4].select_slider("Auto safe trade",["Off","On"])

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

# async def get_prediction(session, input_data, auth_token):
#     """
#     Get a prediction from the ML service
#     :param session: aiohttp.ClientSession - an aiohttp session object
#     :param input_data: str - the input data for the prediction
#     :param auth_token: str - the auth token to be used for authentication
#     :return: dict - the ML service's response
#     """
#     ml_url = 'http://ml-service:5000/predict'
#     headers = {'Authorization': auth_token}
#     ml_data = {'input': input_data}
#     async with session.post(ml_url, json=ml_data, headers=headers) as ml_response:
#         ml_json = await ml_response.json()
#     return ml_json

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