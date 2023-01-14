import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(''), '..'))) # 
import streamlit as st
from  data_eng import data_util

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

col1, col2, col3 = st.columns([7,5,4])
col2.markdown("### WasiuNet")
col = st.columns([1,5,5,5,2,1])
asset1 = col[1].selectbox("select asset",["btc-usd", "eth-usd"])
trade_date = col[2].date_input("start safe entry search - date")
trade_time = col[3].time_input("start safe entry search - time")
col = col[4].select_slider("Auto safe trade",["Off","On"])
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