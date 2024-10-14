import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

# Function to display content based on the sidebar section selection
def show_section(section):
    if section == "Dataset and Competition Info":
        st.subheader("Dataset Description")
        st.write("""
        **The Healthy Brain Network (HBN)** dataset is a clinical sample of about five-thousand 5-22 year-olds 
        who have undergone both clinical and research screenings. The objective of the HBN study is to find 
        biological markers that will improve the diagnosis and treatment of mental health and learning disorders 
        from an objective biological perspective. Two elements of this study are being used for this competition: 
        physical activity data (wrist-worn accelerometer data, fitness assessments and questionnaires) and 
        internet usage behavior data. The goal of this competition is to predict a participant's Severity 
        Impairment Index (SII), a standard measure of problematic internet use.
        
        Note that this is a Code Competition, in which the actual test set is hidden. In this public version, 
        we give some sample data in the correct format to help you author your solutions. The full test set 
        comprises about 3800 instances.

        The competition data includes various instruments:
        - **Demographics**: Age and sex of participants
        - **Internet Use**: Number of hours using computer/internet per day
        - **Children's Global Assessment Scale (CGAS)**: Rating general functioning of youths under 18
        - **Physical Measures**: Blood pressure, heart rate, height, weight, waist/hip measurements
        - **Fitness Assessments**: Cardiovascular fitness, muscular strength, endurance, flexibility, body composition
        - **Bio-electric Impedance Analysis (BIA)**: Body composition elements including BMI, fat, muscle, water
        - **Sleep Disturbance Scale**: Sleep disorder categorization
        - **Actigraphy**: Objective physical activity measurement via a biotracker
        - **Parent-Child Internet Addiction Test (PCIAT)**: Scale measuring internet addiction traits

        For more details, check the Kaggle competition or the Colab notebook:
        - [Kaggle Competition](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use)
        - [Colab Notebook](https://colab.research.google.com/drive/123G-Y8jpjPBaExsx1Ocdw1LBgruZhq94?usp=sharing )
        """)

    elif section == "EDA & Preprocessing":
        st.subheader("EDA & Preprocessing")
        st.write("""
        **Exploratory Data Analysis (EDA)** involves examining data patterns and relationships:
        - Boxplots to assess the distribution of variables and their relationship with SII.
        - Correlation heatmaps to observe relationships between variables.
        
        **Preprocessing Steps**:
        - Handling missing values by replacing them with the mean.
        - Encoding categorical variables such as seasons.
        - Scaling features using StandardScaler to normalize the data.
        - Applying PCA to reduce dimensions and keep 95% of the variance.
        """)

        # Displaying example images (use your actual images' paths)
        st.image("C:\\Users\\ruban\\Downloads\\ml\\1.png", caption="Boxplot of SII vs Categorical Variables", use_column_width=True)
        st.image("C:\\Users\\ruban\\Downloads\\ml\\2.png", caption="Correlation Heatmap", use_column_width=True)
        st.image("C:\\Users\\ruban\\Downloads\\ml\\3.png", caption="PCA Transformation", use_column_width=True)

    elif section == "Model Evaluation":
        st.subheader("Model Evaluation")
        evaluation_data = {
            "Model": ["Random Forest", "Logistic Regression", "SVM", "K-Nearest Neighbors", 
                      "Decision Tree", "Naive Bayes", "Gradient Boosting"],
            "Accuracy": [0.714646, 0.713384, 0.708333, 0.705808, 0.651515, 0.674242, 0.696970]
        }
        evaluation_df = pd.DataFrame(evaluation_data)
        st.write("### Model Evaluation")
        st.table(evaluation_df)
        st.image("C:\\Users\\ruban\\Downloads\\ml\\model performance.png", caption="PCA Transformation", use_column_width=True)

# ---- MAIN APP ----
st.title('SII Prediction Using Pre-trained Model')

# Initialize session state to handle section navigation
if "section" not in st.session_state:
    st.session_state.section = "Introduction"

# ---- SIDEBAR ----
st.sidebar.title("Navigation")
# Create buttons in the sidebar to navigate between sections
if st.sidebar.button("Dataset and Competition Info"):
    st.session_state.section = "Dataset and Competition Info"
if st.sidebar.button("EDA & Preprocessing"):
    st.session_state.section = "EDA & Preprocessing"
if st.sidebar.button("Model Evaluation"):
    st.session_state.section = "Model Evaluation"
if st.sidebar.button("Prediction"):
    st.session_state.section = "Prediction"

# Show the section based on the sidebar button clicked
show_section(st.session_state.section)

# ---- MAIN SECTION ----
if st.session_state.section == "Prediction":
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
