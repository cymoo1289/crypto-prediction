import streamlit as st
import pandas as pd
from pathlib import Path
import yaml
#import tensorflow 
from PIL import Image
import streamlit_authenticator as stauth 
from datetime import time
from streamlit_option_menu import option_menu
import database
import model 
import time
import altair as alt

def feedback_page(db):
    st.title("Feedback")

    d = st.date_input("Today's date",None, None, None, None)
    
    feedback_input = st.text_input("Please give feedbacks")
    if st.button("Submit"):
        try:
            db.insert_feedback(feedback_input)
            st.success("Feedback submitted")
        except:
            st.warning("Something went wrong, cannot submit feedback")
    feedbacks = db.get_feedback()
    print(feedbacks)

def pw_reset_page():
    if authentication_status:
        try:
            if authenticator.reset_password(username, 'Reset password'):
                with open('config.yaml', 'w') as file:
                    config['credentials'] = authenticator.credentials
                    yaml.dump(config, file, default_flow_style=False)
                    print(config)
                st.success('Password modified successfully')
        except Exception as e:
            st.error(e)

def register_page():
    st.title("Register")
    try:
        if authenticator.register_user('Register user', preauthorization = False):
            with open('config.yaml', 'w') as file:
                config['credentials'] = authenticator.credentials
                yaml.dump(config, file, default_flow_style=False)
                print(config)
            st.success('User registered successfully')
    except Exception as e:
        st.error(e)

def data_analysis_page():
    st.title("Data Analysis")
    model_btc = model.LSTM_MODEL("BTC","models/lstm_model_2.h5")
    #df = model_btc.get_yahoo_data()
    # import pandas_datareader as web
    # df = web.DataReader("BTC-USD", data_source = 'yahoo', start = "2018-01-01" ,end = "2022-12-03")
    # print(df)
    model_btc.get_lstm_model()
    #df = df.filter(['Close'])
    #print(df.columns)
    #st.line_chart(data = df, y = "Close")

    if st.button("Predict"):
        preds= model_btc.predict_result()[0][0] 
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.05)
            my_bar.progress(percent_complete + 1)
        st.write("The predicted next day closing at {} $USD".format(round(preds,2)))


def home_page():
    

    st.title(f'Welcome *{st.session_state["name"]}*')
    st.title("This is a demo application for crypto prediction")    
    

st.set_page_config(page_title="Crypto App", page_icon=":bar_chart:", layout="wide")

#hashed_passwords = stauth.hasher(passwords).generate()
# --- USER AUTHENTICATION ---

path = "crypto_app.db"
db = database.SqliteDB(db_path = path)
creds = db.get_creds()[0][0]

with open(creds) as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)




# try:
#     if authenticator.register_user('Register user', preauthorization = False):
#         print(config)
#         with open('config.yaml', 'w') as file:
#             config['credentials'] = authenticator.credentials
#             yaml.dump(config, file, default_flow_style=False)
#             print(config)
#         st.success('User registered successfully')
# except Exception as e:
#     st.error(e)
print(st.session_state["authentication_status"])
if st.session_state["authentication_status"] is None:
    selected = option_menu(
                menu_title=None,  # required
                options=["Login", "Register"],  # required
                icons=["house", "book"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
                orientation="horizontal",
            )

    if selected == "Login":
        name, authentication_status, username = authenticator.login("Login", "main")
    elif selected == "Register":
        register_page()


if st.session_state["authentication_status"]:
    print(st.session_state)

    #authenticator.logout("Logout", "main")
    with st.sidebar:
        selected = option_menu(
            menu_title = "Main Menu",
            options = ["Home","Data Analysis","Register User","Feedback","Password Reset"],
            default_index = 0
        )

    authenticator.logout('Logout', 'sidebar')

    if selected == "Home":
        home_page()
    elif selected == "Data Analysis":
        data_analysis_page()
    elif selected == "Register User":
        register_page()
    elif selected == "Feedback":
        feedback_page(db) 
    elif selected == "Password Reset":
        pw_reset_page()

elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')

elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')


       



    