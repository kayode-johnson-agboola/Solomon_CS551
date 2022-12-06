import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image


# Load  model a 
model = joblib.load(open("model-v1.joblib","rb"))

def data_preprocessor(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    covid19.state = covid19.state.map({'AL':0,'MP':1,'ME':2,'VA':3,'DE':4,'FSM':5,'CO':6,'MI':7,'NYC':8,'NC':9,'PR':10,'MS':11,'KY':12,'TN':13,'ID':14,'OH':15,'WY':16,'CT':17,'WI':18,'MT':19,'IL':20,'OR':21,'AZ':22,'GA':23,'RMI':24,'UT':25,'MA':26,'OK':27,'NJ':28,'CA':29,'NE':30,})
    return covid19

def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data = data,columns = ['Percentage'],index = ['Low death','Ave death','High death'])
    ax = grad_percentage.plot(kind='barh', figsize=(7, 4), color='#722f37', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel(" Percentage(%) Confidence Level", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Confirmed Death Ratio", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level ', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()
    return

st.write("""
# COVID Prediction ML Web-App 
This app predicts the ** Confirmed Death Ratio **  using **Cases and death features** input via the **side panel** 
""")

#read in wine image and render with streamlit
image = Image.open('Covid19_image.png')
st.image(image, caption='coronavirus',use_column_width=True)

st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe

    """
    state = st.sidebar.selectbox("Select US States",("AL","MP","ME","VA","DE","FSM","CO","MI","NYC","NC","PR","MS","KY","TN","ID","OH","WY","CT","WI","MT","IL","OR","AZ","GA","RMI","UT","MA","OK","NJ","CA","NE"))
    total_cases = st.sidebar.slider('total_cases', 0, 4885289, 2000000)
    confirmed_cases = st.sidebar.slider('confirmed_cases', 0, 4640489, 2500000)
    probable_cases  = st.sidebar.slider('probable_cases', 0, 763762, 343762)
    total_death  = st.sidebar.slider('total_death', 0, 71408, 36008)
    probable_death  = st.sidebar.slider('probable_death', 0, 7889, 3444)
        
    features = {'state': state,
            'total_cases': total_cases,
            'confirmed_cases': confirmed_cases,
            'probable_cases': probable_cases,
            'total_death': total_death,
            'probable_death': probable_death,
            }
    data = pd.DataFrame(features,index=[0])

    return data

user_input_df = get_user_input()
processed_user_input = data_preprocessor(user_input_df)

st.subheader('User Input parameters')
st.write(user_input_df)

prediction = model.predict(processed_user_input)
prediction_proba = model.predict_proba(processed_user_input)

visualize_confidence_level(prediction_proba)