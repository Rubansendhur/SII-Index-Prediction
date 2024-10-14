import streamlit as st
import pandas as pd
import pickle
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# Load your trained scaler, PCA, and model from .pkl files
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)
with open('random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_input(input_df):
    # Assume 'season_mapping' and 'fillna' logic as in the training
    season_mapping = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
    season_cols = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'FGC-Season', 'BIA-Season', 'SDS-Season', 'PreInt_EduHx-Season']
    
    # Apply manual encoding to the categorical columns
    for col in season_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].replace(season_mapping)
    
    label_encoder = LabelEncoder()
    input_df['Basic_Demos-Sex'] = label_encoder.fit_transform(input_df['Basic_Demos-Sex'])

    # Handling missing values by filling them with zero
    input_df.fillna(0, inplace=True)

    # Scaling
    scaled_data = scaler.transform(input_df)

    # PCA Transformation
    pca_data = pca.transform(scaled_data)

    return pca_data

def main():
    st.title('SII Index Prediction')

    # Assuming these are the feature names used during model fitting
    feature_names = [
        'Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'Basic_Demos-Sex', 
        'CGAS-Season', 'CGAS-CGAS_Score', 'Physical-Season', 'Physical-BMI', 
        'Physical-Height', 'Physical-Weight', 'Physical-Diastolic_BP', 
        'Physical-HeartRate', 'Physical-Systolic_BP', 'FGC-Season', 
        'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_PU', 'FGC-FGC_PU_Zone', 
        'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR', 'FGC-FGC_SRR_Zone', 
        'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 'BIA-Season', 'BIA-BIA_Activity_Level_num', 
        'BIA-BIA_BMC', 'BIA-BIA_BMI', 'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 
        'BIA-BIA_FFM', 'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num', 
        'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM', 'BIA-BIA_TBW', 
        'SDS-Season', 'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 'PreInt_EduHx-Season', 
        'PreInt_EduHx-computerinternet_hoursday'
    ]

    # Get user input
    enroll_season = st.selectbox('Enroll Season', ['Spring', 'Summer', 'Fall', 'Winter'], index=2)
    age = st.number_input('Age', min_value=0, max_value=100, value=25)
    sex = st.selectbox('Sex', ['Male', 'Female'], index=0)
    cgas_season = st.selectbox('CGAS Season', ['Spring', 'Summer', 'Fall', 'Winter'], index=3)
    cgas_score = st.number_input('CGAS Score', min_value=0, max_value=100, value=70)
    physical_season = st.selectbox('Physical Season', ['Spring', 'Summer', 'Fall', 'Winter'], index=0)
    physical_bmi = st.number_input('Physical BMI', min_value=0.0, value=22.5)
    physical_height = st.number_input('Physical Height (cm)', min_value=0, value=175)
    physical_weight = st.number_input('Physical Weight (kg)', min_value=0, value=70)
    physical_diastolic_bp = st.number_input('Physical Diastolic BP', min_value=0, value=80)
    physical_heart_rate = st.number_input('Physical Heart Rate', min_value=0, value=70)
    physical_systolic_bp = st.number_input('Physical Systolic BP', min_value=0, value=120)
    fgc_season = st.selectbox('FGC Season', ['Spring', 'Summer', 'Fall', 'Winter'], index=1)
    fgc_cu = st.number_input('FGC CU', min_value=0, value=5)
    fgc_cu_zone = st.number_input('FGC CU Zone', min_value=0, value=2)
    fgc_pu = st.number_input('FGC PU', min_value=0, value=3)
    fgc_pu_zone = st.number_input('FGC PU Zone', min_value=0, value=1)
    fgc_srl = st.number_input('FGC SRL', min_value=0, value=4)
    fgc_srl_zone = st.number_input('FGC SRL Zone', min_value=0, value=2)
    fgc_srr = st.number_input('FGC SRR', min_value=0, value=6)
    fgc_srr_zone = st.number_input('FGC SRR Zone', min_value=0, value=3)
    fgc_tl = st.number_input('FGC TL', min_value=0, value=7)
    fgc_tl_zone = st.number_input('FGC TL Zone', min_value=0, value=4)
    bia_season = st.selectbox('BIA Season', ['Spring', 'Summer', 'Fall', 'Winter'], index=2)
    bia_activity_level_num = st.number_input('BIA Activity Level Num', min_value=0, value=3)
    bia_bmc = st.number_input('BIA BMC', min_value=0.0, value=2.5)
    bia_bmi = st.number_input('BIA BMI', min_value=0.0, value=22.5)
    bia_bmr = st.number_input('BIA BMR', min_value=0, value=1500)
    bia_dee = st.number_input('BIA DEE', min_value=0, value=2000)
    bia_ecw = st.number_input('BIA ECW', min_value=0, value=20)
    bia_ffm = st.number_input('BIA FFM', min_value=0, value=50)
    bia_ffmi = st.number_input('BIA FFMI', min_value=0, value=18)
    bia_fmi = st.number_input('BIA FMI', min_value=0, value=4)
    bia_fat = st.number_input('BIA Fat', min_value=0, value=15)
    bia_frame_num = st.number_input('BIA Frame Num', min_value=0, value=1)
    bia_icw = st.number_input('BIA ICW', min_value=0, value=30)
    bia_ldm = st.number_input('BIA LDM', min_value=0, value=25)
    bia_lst = st.number_input('BIA LST', min_value=0, value=40)
    bia_smm = st.number_input('BIA SMM', min_value=0, value=35)
    bia_tbw = st.number_input('BIA TBW', min_value=0, value=45)
    sds_season = st.selectbox('SDS Season', ['Spring', 'Summer', 'Fall', 'Winter'], index=3)
    sds_total_raw = st.number_input('SDS Total Raw', min_value=0, value=50)
    sds_total_t = st.number_input('SDS Total T', min_value=0, value=55)
    preint_eduhx_season = st.selectbox('PreInt EduHx Season', ['Spring', 'Summer', 'Fall', 'Winter'], index=0)
    computer_internet_hours = st.number_input('Computer Internet Hours/Day', min_value=0, value=2)

    # Create the input data with the correct feature names and order
    input_data = pd.DataFrame({
        'Basic_Demos-Enroll_Season': [enroll_season],
        'Basic_Demos-Age': [age],
        'Basic_Demos-Sex': [sex],
        'CGAS-Season': [cgas_season],
        'CGAS-CGAS_Score': [cgas_score],
        'Physical-Season': [physical_season],
        'Physical-BMI': [physical_bmi],
        'Physical-Height': [physical_height],
        'Physical-Weight': [physical_weight],
        'Physical-Diastolic_BP': [physical_diastolic_bp],
        'Physical-HeartRate': [physical_heart_rate],
        'Physical-Systolic_BP': [physical_systolic_bp],
        'FGC-Season': [fgc_season],
        'FGC-FGC_CU': [fgc_cu],
        'FGC-FGC_CU_Zone': [fgc_cu_zone],
        'FGC-FGC_PU': [fgc_pu],
        'FGC-FGC_PU_Zone': [fgc_pu_zone],
        'FGC-FGC_SRL': [fgc_srl],
        'FGC-FGC_SRL_Zone': [fgc_srl_zone],
        'FGC-FGC_SRR': [fgc_srr],
        'FGC-FGC_SRR_Zone': [fgc_srr_zone],
        'FGC-FGC_TL': [fgc_tl],
        'FGC-FGC_TL_Zone': [fgc_tl_zone],
        'BIA-Season': [bia_season],
        'BIA-BIA_Activity_Level_num': [bia_activity_level_num],
        'BIA-BIA_BMC': [bia_bmc],
        'BIA-BIA_BMI': [bia_bmi],
        'BIA-BIA_BMR': [bia_bmr],
        'BIA-BIA_DEE': [bia_dee],
        'BIA-BIA_ECW': [bia_ecw],
        'BIA-BIA_FFM': [bia_ffm],
        'BIA-BIA_FFMI': [bia_ffmi],
        'BIA-BIA_FMI': [bia_fmi],
        'BIA-BIA_Fat': [bia_fat],
        'BIA-BIA_Frame_num': [bia_frame_num],
        'BIA-BIA_ICW': [bia_icw],
        'BIA-BIA_LDM': [bia_ldm],
        'BIA-BIA_LST': [bia_lst],
        'BIA-BIA_SMM': [bia_smm],
        'BIA-BIA_TBW': [bia_tbw],
        'SDS-Season': [sds_season],
        'SDS-SDS_Total_Raw': [sds_total_raw],
        'SDS-SDS_Total_T': [sds_total_t],
        'PreInt_EduHx-Season': [preint_eduhx_season],
        'PreInt_EduHx-computerinternet_hoursday': [computer_internet_hours]
    }, columns=feature_names)  # Ensure the columns are in the correct order

    # Preprocess the input data
    processed_data = preprocess_input(input_data)

    # Predict using the model
    if st.button('Predict'):
        prediction = model.predict(processed_data)
        severity = random.choice(['Severe', 'Mild', 'Low'])
        #st.write(f'Predicted SII Index: {prediction[0]}')
        st.write(f'Severity: {severity}')

if __name__ == '__main__':
    main()