import streamlit as st
import pandas as pd
from pathlib import Path
import yaml
#import tensorflow 
from PIL import Image
import streamlit_authenticator as stauth 
from datetime import time
from streamlit_option_menu import option_menu

def feedback_page():
    st.title("Feedback")

    d = st.date_input("Today's date",None, None, None, None)

    question_1 = st.text_input("Please give feedbacks")

def register_page():
    st.title("Register")

def data_analysis_page():
    st.title("Data Analysis")

def home_page():
    #authenticator.logout('Logout', 'main')

    st.write(f'Welcome *{st.session_state["name"]}*')
        
    st.title('Some content')
    input = st.slider(
    "Choose length of predictions:",
    value=(time(11, 30), time(12, 45)))
    st.write("You've chosen length of prediction for:", input)
    image = Image.open('scripts/prediction.jpg')

    st.image(image, caption='BTC predictions')

st.set_page_config(page_title="Crypto App", page_icon=":bar_chart:", layout="wide")

#hashed_passwords = stauth.hasher(passwords).generate()
# --- USER AUTHENTICATION ---

with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)


name, authentication_status, username = authenticator.login("Login", "main")

if st.session_state["authentication_status"]:
    authenticator.logout("Logout", "main")
    with st.sidebar:
        selected = option_menu(
            menu_title = "Main Menu",
            options = ["Home","Data Analysis","Register User","Feedback"],
            default_index = 0
        )
    if selected == "Home":
        home_page()
    elif selected == "Data Analysis":
        data_analysis_page()
    elif selected == "Register User":
        register_page()
    elif selected == "Feedback":
        feedback_page() 

elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')

elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')

    