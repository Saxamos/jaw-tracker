import streamlit as st
import pandas as pd
from io import StringIO

st.title("Welcome in Jaw Tracker 🦈")

uploaded_file = st.file_uploader(
    """
    Upload a panoramic radio 🦷
    We'll see if that match something in our DB!
    """
)
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
