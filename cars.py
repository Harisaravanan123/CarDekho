import  pandas as pd
import pickle
import base64
import streamlit as st
import numpy as np
st.set_page_config(page_title="car dekho",page_icon=":car:",layout="centered")
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg("cardekho image.webp")

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #9899AA;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color:maroon;'>Car Dekho - Cars Price Prediction</h1>", unsafe_allow_html=True)
with open('OHE.pkl','rb')as encoder1:
    onehot = pickle.load(encoder1)
with open('OE.pkl','rb')as encoder2:
    ordinal = pickle.load(encoder2)  
with open('rscv_m.pkl','rb')as model:
    algo = pickle.load(model)   
df = pd.read_csv('data_generated.csv')
num = df.select_dtypes(include=['int64','float']).columns
for col in num:
    df[col]=df[col].astype('float')
obj = df.select_dtypes(include=['object']).columns
for col in obj:
    df[col] = df[col].astype('category')    

    
# User Input
st.sidebar.header("Car Features")    
City = st.sidebar.selectbox(label='City',options=df['City'].unique()) 
filtered_city = df[df['City']==City]
oem_options = sorted(filtered_city['oem'].unique())
Manufacturer = st.sidebar.selectbox(label='Manufacturer',options=oem_options)
filtered_oem = filtered_city[filtered_city['oem']==Manufacturer]
car_age_options = sorted(filtered_oem['Car age'].unique())
carage = st.sidebar.selectbox(label='Car Age',options=car_age_options)
filtered_carage = filtered_oem[filtered_oem['Car age']==carage]
modelyear_opt = sorted(filtered_carage['modelYear'].unique())
modelyear = st.sidebar.selectbox(label='ModelYear',options=modelyear_opt)
filtered_mdyear = filtered_carage[filtered_carage['modelYear']==modelyear]
Insurance_validity_opt = sorted(filtered_mdyear['Insurance Validity'].unique())
insurancevalidity = st.sidebar.selectbox(label='Insurance Validity',options=Insurance_validity_opt)
filtered_Iv = filtered_mdyear[filtered_mdyear['Insurance Validity']==insurancevalidity]
fueltype_options = sorted(filtered_Iv['ft'].unique())
fueltype = st.sidebar.selectbox(label='fueltype',options=fueltype_options)
filtered_ft = filtered_Iv[filtered_Iv['ft']==fueltype]
bt_options = sorted(filtered_ft['bt'].unique())
body_type = st.sidebar.selectbox(label='bodytype',options=bt_options)
filtered_bt = filtered_ft[filtered_ft['bt']==body_type]
km_options = sorted(filtered_bt['km'].unique())
kilometer_runned = st.sidebar.selectbox(label='kilometer runned',options=km_options)
filtered_km = filtered_bt[filtered_bt['km']==kilometer_runned]
transmission_options = sorted(filtered_km['transmission'].unique())
transmission = st.sidebar.selectbox(label='transmission',options=transmission_options)
filtered_transmission = filtered_km[filtered_km['transmission']==transmission]
seats_options = sorted(filtered_transmission['Seats'].unique())
seats = st.sidebar.selectbox(label='seats',options=seats_options)
filtered_seats = filtered_transmission[filtered_transmission['Seats']==seats]
ownership_options = sorted(filtered_seats['Ownership'].unique())
ownership = st.sidebar.selectbox(label='ownership',options=ownership_options)
filtered_ownership = filtered_seats[filtered_seats['Ownership']==ownership]
ed_options = sorted(filtered_ownership['Engine Displacement'].unique())
engine_displacement = st.sidebar.selectbox(label='engine displacement',options=ed_options)
filterd_ed = filtered_ownership[filtered_ownership['Engine Displacement']==engine_displacement]
mileage_options = sorted(filterd_ed['Mileage'].unique())
mileage = st.sidebar.selectbox(label='mileage',options=mileage_options)
filtered_mileage = filterd_ed[filterd_ed['Mileage']==mileage]
max_power_options = sorted(filtered_mileage['Max Power'].unique())
max_power = st.sidebar.selectbox(label='max_power',options=max_power_options)

encoded_ordinal = ordinal.transform([[ownership,insurancevalidity]])[0]
ins_encoded = encoded_ordinal[1]
ownership_encoded = encoded_ordinal[0]
numerical_inputs = [
    float(kilometer_runned),
    float(modelyear),
    float(engine_displacement),
    float(mileage),
    float(max_power),
    float(carage)
]
numerical_inputs = np.array(numerical_inputs).reshape(1,-1)
onehot_columns = [['fueltype', 'body_type', 'transmission', 'Manufacturer','seats','City']]
onehot_encoded = onehot.transform(onehot_columns).flatten().reshape(1,-1)
final_input = [
    np.array([numerical_inputs[0][0]]),  # kilometer_runned
    np.array([numerical_inputs[0][1]]),  # modelyear
    np.array([ins_encoded]),
    np.array([ownership_encoded]),
    np.array([numerical_inputs[0][2]]),  # engine_displacement
    np.array([numerical_inputs[0][3]]),  # mileage
    np.array([numerical_inputs[0][4]]),  # max_power
    np.array([numerical_inputs[0][5]]),  # carage
    onehot_encoded.flatten()  # Flatten the one-hot encoded features
]
final_array = np.hstack(final_input).reshape(1,-1)
button = st.sidebar.button('Predict Price')
if button:
    prediction = algo.predict(final_array)[0]
    price_r = round(prediction,2)
    
    st.markdown(f'<h2 style="text-align: center;color:Darkgreen;">Predicted Car Price : ₹ {price_r} </h2>', unsafe_allow_html=True)

















