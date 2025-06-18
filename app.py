import streamlit as st
import pandas as pd
import numpy as np
import joblib 
from joblib import load
import pickle 
import sys



### Making all text white 
st.markdown("""
    <style>
    /* Make all paragraph text white */
    .stApp {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
### Getting a background
import base64

# Function to load and encode image
def load_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

# Function to apply image as background
def background_image_style(path):
    encoded = load_image(path)
    return f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)),
                          url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """

# Apply background image
st.markdown(background_image_style("climbing everest.jpg"), unsafe_allow_html=True)
# Add text as title with specific features
st.markdown("""
    <h2 style='text-align: center; color: #FFFFFF; font-family: Georgia;'>
        Welcome to your expedition to the Himalayas! 🏔️
    </h2>
""", unsafe_allow_html=True)

st.markdown("<h3 style='color: white; font-family: Georgia;'>Tell us a bit about yourself and we'll recommend the perfect mountain for you to climb according to your profile.</h3>", unsafe_allow_html=True)

#st.image('climbing everest.jpg', caption="This could be you", use_container_width=True)

## Change color to text input 
st.markdown("""
    <style>
    label {
        color: white !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


age = st.text_input("How old are you?")
try:
    age = int(age)
except:
    #st.error("Please make sure that you only enter a number")
    st.stop()

nb_members = st.text_input("How many people will join the expedition?")
try:
    nb_members = int(nb_members)
except:
    st.stop()

nb_hired = st.text_input("How many of those are hired staff?")
try:
    nb_hired = int(nb_hired)
except:
    st.stop()

pct_hired= int(nb_hired)/int(nb_members)


season_list = ["Spring", "Summer", "Autumn", "Winter"]
with st.container(border=True):
    season = st.selectbox("In which season would you like to hike?", season_list)
    
sex_list = ["M", "F"]
with st.container(border=True):
    sex = st.selectbox("Sex", sex_list)
o2_list = ["Yes", "No"]
with st.container(border=True):
    o2 = st.selectbox("Will you bring oxygen?", o2_list)
if o2 == "Yes":
    o2used = 1
if o2 == "No":
    o2used = 0

difficulty_list = ["Easy","Medium", "Hard" ]
with st.container(border=True):
    difficulty = st.selectbox ("Difficulty level", difficulty_list)
# Here we need to define the categories with the ranges defined by Diogo

Country = ['Other',
 'Afghanistan',
 'Albania',
 'Algeria',
 'Andorra',
 'Argentina',
 'Armenia',
 'Australia',
 'Austria',
 'Azerbaijan',
 'Bahrain',
 'Bangladesh',
 'Belarus',
 'Belgium',
 'Bhutan',
 'Bolivia',
 'Bosnia-Herzegovina',
 'Botswana',
 'Brazil',
 'Bulgaria',
 'Canada',
 'Chile',
 'China',
 'Colombia',
 'Costa Rica',
 'Croatia',
 'Cuba',
 'Cyprus',
 'Czech Republic',
 'Czechoslovakia',
 'Denmark',
 'Dominica',
 'Dominican Republic',
 'Ecuador',
 'Egypt',
 'El Salvador',
 'Estonia',
 'Finland',
 'France',
 'Georgia',
 'Germany',
 'Greece',
 'Guatemala',
 'Honduras',
 'Hungary',
 'Iceland',
 'India',
 'Indonesia',
 'Iran',
 'Iraq',
 'Ireland',
 'Israel',
 'Italy',
 'Japan',
 'Jordan',
 'Kazakhstan',
 'Kenya',
 'Kosovo',
 'Kuwait',
 'Kyrgyz Republic',
 'Latvia',
 'Lebanon',
 'Libya',
 'Liechtenstein',
 'Lithuania',
 'Luxembourg',
 'Macedonia',
 'Malaysia',
 'Malta',
 'Mauritius',
 'Mexico',
 'Moldova',
 'Mongolia',
 'Montenegro',
 'Morocco',
 'Myanmar',
 'Nepal',
 'Netherlands',
 'New Zealand',
 'Norway',
 'Oman',
 'Pakistan',
 'Palestine',
 'Panama',
 'Paraguay',
 'Peru',
 'Philippines',
 'Poland',
 'Portugal',
 'Qatar',
 'Romania',
 'Russia',
 'San Marino',
 'Saudi Arabia',
 'Serbia',
 'Singapore',
 'Slovakia',
 'Slovenia',
 'South Africa',
 'South Korea',
 'Spain',
 'Sri Lanka',
 'Sweden',
 'Switzerland',
 'Syria',
 'Taiwan',
 'Tajikistan',
 'Tanzania',
 'Thailand',
 'Tunisia',
 'Turkey',
 'UAE',
 'UK',
 'USA',
 'Ukraine',
 'Uruguay',
 'Uzbekistan',
 'Venezuela',
 'Vietnam',
 'Yugoslavia']
   
with st.container(border=True):
    country = st.selectbox("Country", Country)
# Function to get country_max_height  
df= pd.read_csv('Country_max_heigth_list.csv')
def get_highest_peak(country):
    country = country.lower()
    if country in df['Country'].str.lower().values:
        peak = df[df['Country'].str.lower() == country]['Highest_Peak_m']
    else:
        peak = df[df['Country'].str.lower() == 'other']['Highest_Peak_m']
    if not peak.empty:
        return peak.values[0]
country_max_height = get_highest_peak(country)

# Definition of new data for model 1 

new_data = pd.DataFrame({
    'mseason': [season],
    'sex': [sex],
    'country_max_height': [country_max_height],
    'mo2used': [o2used],
    'nb_members': [nb_members],
    'pct_hired': [pct_hired],
    'age': [age],
})

# Here we need to run model 1 to get max_height

scaler = load('scaler.joblib')
encoder = load('encoder (1).joblib')
model = load('model.joblib')

new_data_num = scaler.transform(new_data.select_dtypes(include="number"))
new_data_cat = encoder.transform(new_data.select_dtypes(exclude="number"))

new_data_scaled = pd.concat([new_data_num,new_data_cat], axis=1)                                                   

max_height_prediction = float(model.predict(new_data_scaled)[0])

#maxheight = joblib.load('max_height_full_pipeline.pkl')
#max_height_prediction = maxheight.predict(new_data)

st.write(f"According to our analysis you will be able to climb up to {max_height_prediction} meters!")

#  Definition of new data for model 2

new_data = pd.DataFrame({
    'mseason': [season],
    'sex': [sex],
    'country_max_height': [country_max_height],
    'mo2used': [o2used],
    'nb_members': [nb_members],
    'pct_hired': [pct_hired],
    'age': [age],
    'peakid': []
})
#st.write(country_max_height)
