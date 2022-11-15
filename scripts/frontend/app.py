import streamlit as st
import pandas as pd

st.title("Cryptocurreny Prediction Demo")
st.write("Showing **XRP-USD** Crypto Price Graph")
df_ori = pd.read_csv("../../data/XRP-USD.csv")
df1 = df_ori[['Date','Close']]
df2 = df_ori[['Date','Volume']]
st.slider(label="slide")
st.header("XRP-USD Close")
st.line_chart(df1,x = "Date", y="Close")

st.header("XRP-USD Volume")
st.line_chart(df2,x = "Date", y="Volume")
