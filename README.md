SII Prediction System for Internet Usage Behavior

Project Overview
This project aims to predict the Severity Impairment Index (SII) based on internet usage behavior and other health-related data from the Healthy Brain Network (HBN) dataset. The SII is a measure of problematic internet use, and this system leverages machine learning to provide insights into the severity level of internet addiction.

Key Features:
Predicts the SII level: Low, Mild, Severe, Heavy Severe
Utilizes various health metrics such as physical activity, demographics, sleep patterns, and internet usage behavior.
Provides business insights for clinics, insurance companies, schools, and tech firms.
Pre-trained model (Random Forest) included for prediction.
Interactive prediction system built using Streamlit.

├── app.py                   # Streamlit app for SII prediction
├── random_forest.pkl         # Pre-trained Random Forest model
├── scaler.pkl                # Saved Scaler used for data preprocessing
├── pca.pkl                   # PCA model for dimensionality reduction
├── requirements.txt          # Python packages required to run the project
├── data/                     # Folder containing dataset (train.csv and test.csv)
├── images/                   # EDA and preprocessing images for insights
├── README.md                 # Project documentation (this file)


Dataset Description
The dataset is from the Healthy Brain Network (HBN), which includes around 5,000 participants aged 5-22. It contains various physical and internet usage metrics, such as:

Demographics: Age, Sex
Physical Measures: BMI, heart rate, height, weight
Internet Usage: Hours spent online per day
Children's Global Assessment Scale (CGAS): Functioning levels for youths
Bio-electric Impedance Analysis (BIA): Body composition data
Sleep Disturbance Scale (SDS): Sleep-related issues
Parent-Child Internet Addiction Test (PCIAT): Used to assess internet addiction levels.
Links:
Kaggle Dataset: Healthy Brain Network Dataset
Colab Notebook: Colab Link
How to Run the Project
Requirementspip install -r requirements.txt

Ensure you have Python 3.7+ and install the dependencies listed in the requirements.txt file:

pip install -r requirements.txt


Steps to Run Locally
Clone the repository:

git clone https://github.com/your-username/sii-prediction.git
cd sii-prediction

Install dependencies:
pip install -r requirements.txt

Run the Streamlit App:
streamlit run app.py


Upload CSV: Once the app is running, upload a CSV file with the required features to get the predicted SII level.

Note: The required columns for the CSV include:
Basic_Demos-Enroll_Season, Basic_Demos-Age, Basic_Demos-Sex, CGAS-Season, CGAS-CGAS_Score, Physical-Season, Physical-BMI, Physical-Height, Physical-Weight, etc.
Exploratory Data Analysis and Preprocessing
In this project, we conducted an Exploratory Data Analysis (EDA) and Preprocessing phase that included:

Handling missing data by filling missing values with the mean.
Encoding categorical variables (e.g., Season) using manual mappings.
Scaling features using StandardScaler.
Dimensionality reduction using PCA to retain 95% of the variance.
Sample EDA Images are provided in the images/ directory.

Modeling
The following models were trained and evaluated using the SII prediction data:

Random Forest Classifier (chosen for final model)
Logistic Regression
SVM
K-Nearest Neighbors
Decision Tree
Naive Bayes
Gradient Boosting
Model accuracy results:

yaml
Copy code
Random Forest: 71.5%
Logistic Regression: 71.3%
SVM: 70.8%
K-Nearest Neighbors: 70.5%
Decision Tree: 65.1%
Naive Bayes: 67.4%
Gradient Boosting: 69.7%
Business Insights
The system can provide insights to:

Clinics: Offer personalized therapy for internet addiction.
Insurance Providers: Create specialized health insurance plans.
Schools: Provide wellness workshops to students on responsible internet use.
Tech Companies: Offer internet monitoring tools for parents.
Research Institutions: License the data for studying internet addiction.
Contact Information
For any questions or feedback, please contact:

Email: Rubansendhur78409@gmail.com
GitHub: RubanSendhur



