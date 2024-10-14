import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Provide links to Kaggle and Colab notebooks
st.markdown('[Go to Kaggle](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use)', unsafe_allow_html=True)
st.markdown('[Go to Colab Notebook](https://colab.research.google.com/drive/123G-Y8jpjPBaExsx1Ocdw1LBgruZhq94?usp=sharing )', unsafe_allow_html=True)


# Function to load saved model, scaler, and PCA components
def load_artifacts():
    with open('random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    return model, scaler, pca

# Preprocess uploaded data without one-hot encoding (use manual encoding as in training)
def preprocess_data(uploaded_df, features, scaler, pca):
    # Select the required features
    X = uploaded_df[features]

    # Define mappings for categorical columns (as used during training)
    season_mapping = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
    X['Basic_Demos-Enroll_Season'] = X['Basic_Demos-Enroll_Season'].map(season_mapping)
    X['CGAS-Season'] = X['CGAS-Season'].map(season_mapping)
    X['Physical-Season'] = X['Physical-Season'].map(season_mapping)
    X['FGC-Season'] = X['FGC-Season'].map(season_mapping)
    X['BIA-Season'] = X['BIA-Season'].map(season_mapping)
    X['SDS-Season'] = X['SDS-Season'].map(season_mapping)
    X['PreInt_EduHx-Season'] = X['PreInt_EduHx-Season'].map(season_mapping)

    # Handling missing values by filling with the mean (for numeric columns)
    X.fillna(X.mean(), inplace=True)

    # Feature Scaling
    X_scaled = scaler.transform(X)

    # Apply PCA
    X_pca = pca.transform(X_scaled)
    
    return X_pca

# Mapping numerical predictions to the actual labels
sii_labels = {0: 'Low', 1: 'Mild', 2: 'Severe', 3: 'Heavy Severe'}

# Predict function
def make_predictions(model, X_pca):
    predictions = model.predict(X_pca)
    predicted_labels = [sii_labels[pred] for pred in predictions]  # Map numerical predictions to categorical labels
    return predicted_labels

# Streamlit Application Layout
st.title('SII Prediction Using Pre-trained Model')

# Load pre-trained artifacts
model, scaler, pca = load_artifacts()

# File upload widget
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load uploaded file into a pandas DataFrame
    uploaded_df = pd.read_csv(uploaded_file)
    
    st.write("File uploaded successfully!")
    
    # Display the uploaded data
    if st.checkbox('Show uploaded data'):
        st.write(uploaded_df)
    
    # Automatically select features that were used during model training
    st.write("Automatically selecting features used for training...")
    features = [
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

    if st.button('Make Predictions'):
        # Preprocess the uploaded data
        X_pca = preprocess_data(uploaded_df, features, scaler, pca)
        
        # Make predictions
        predicted_labels = make_predictions(model, X_pca)
        
        # Add predictions to the DataFrame
        uploaded_df['Predicted_SII'] = predicted_labels
        
        # Display predictions
        st.write("Predictions:")
        st.write(uploaded_df[['Predicted_SII']])

        # Optionally show additional evaluation metrics if the true labels are present in the uploaded file
        if 'sii' in uploaded_df.columns:  # Assuming 'sii' is the target column in the uploaded file
            true_labels = uploaded_df['sii'].dropna()  # Drop NaN values from true_labels
            predicted_filtered = uploaded_df['Predicted_SII'].loc[uploaded_df['sii'].notnull()]  # Keep only rows where `sii` is not NaN
            
            # Ensure true_labels and predicted_filtered have the same length before evaluation
            if len(true_labels) == len(predicted_filtered):
                accuracy = accuracy_score(true_labels, [list(sii_labels.keys())[list(sii_labels.values()).index(pred)] for pred in predicted_filtered])
                st.write(f"Model Accuracy: {accuracy}")
                st.write("Classification Report:")
                st.text(classification_report(true_labels, [list(sii_labels.keys())[list(sii_labels.values()).index(pred)] for pred in predicted_filtered]))
                st.write("Confusion Matrix:")
                st.write(confusion_matrix(true_labels, [list(sii_labels.keys())[list(sii_labels.values()).index(pred)] for pred in predicted_filtered]))
            else:
                st.write("Mismatch in length between true labels and predictions after filtering NaN values.")
