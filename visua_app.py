import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


# NumPy for numerical computing
import numpy as np
# Pandas for DataFrames
import pandas as pd
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score,classification_report, roc_curve, auc
from sklearn.metrics import accuracy_score
# Load data
df1 = pd.read_csv('Survival.csv')
cols = ['Treated_with_drugs', 'Patient_Age', 'Patient_Body_Mass_Index', 'Patient_Smoker', 'Patient_Rural_Urban', 'Patient_mental_condition', 'A', 'B', 'C', 'D', 'E', 'F', 'Z', 'Number_of_prev_cond', 'Survived_1_year']
df1 = df1[df1.Patient_Smoker != 'Cannot say']
df1['Number_of_prev_cond'] = df1['Number_of_prev_cond'].fillna((df1['Number_of_prev_cond'].median()))
for column in df1.columns:
    most_frequent_value = df1[column].mode()[0]
    df1[column] = df1[column].fillna(most_frequent_value)

numer_columns = ['Patient_Age', 'Patient_Body_Mass_Index', 'A', 'B', 'C', 'D', 'E', 'F', 'Number_of_prev_cond','Survived_1_year']
df2 = df1[numer_columns]

# Define functions for each visualization
def correlation_heatmap():
    # Calculate the correlation matrix
    correlation_matrix = df2.corr()

    # Create the heatmap using Plotly
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='Viridis',
        colorbar=dict(title='Correlation')
    ))

    # Add text annotations
    for i, row in enumerate(correlation_matrix.values):
        for j, value in enumerate(row):
            heatmap_fig.add_annotation(
                x=correlation_matrix.columns[j],
                y=correlation_matrix.index[i],
                text=f'{value:.2f}',
                showarrow=False,
                font=dict(size=10, color='white')
            )

    # Display the heatmap
    st.plotly_chart(heatmap_fig)

def age_bin_survival():
    # Define the age bins
    age_bins = [0, 20, 30, 40, 50, 60, 70]

    # Create a new column 'Age_Bin' based on the 'Patient_Age' column
    df1['Age_Bin'] = pd.cut(df1['Patient_Age'], bins=age_bins)

    # Group by age bins and calculate the mean survival rate for each bin
    age_bin_survival = df1.groupby(['Age_Bin', 'Patient_Smoker'])['Survived_1_year'].mean(numeric_only=True).reset_index()

    # Convert 'Age_Bin' to strings just for the plot
    age_bin_survival['Age_Bin'] = age_bin_survival['Age_Bin'].astype(str)

    # Create the age bin survival bar plot using Plotly
    age_bin_fig = px.bar(
        age_bin_survival,
        x='Age_Bin',
        y='Survived_1_year',
        color='Patient_Smoker',
        title='Survival Rate for Different Age Bins',
        labels={'Age_Bin': 'Age Bin', 'Survived_1_year': 'Survival Rate'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    # Add text annotations with the survival rate percentages
    age_bin_fig.update_traces(texttemplate='%{y:.2%}', textposition='outside')

    # Display the age bin survival bar plot
    st.plotly_chart(age_bin_fig)

def drug_survival():
    # Group by drugs and calculate the mean survival rate for each drug
    drug_survivor = df1.groupby(['Treated_with_drugs','Patient_Smoker'])['Survived_1_year'].mean(numeric_only=True).reset_index()

    # Create the drug survival bar plot using Plotly
    drug_fig = px.bar(
        drug_survivor,
        x='Treated_with_drugs',
        y='Survived_1_year',
        color='Patient_Smoker',
        title='Survival Rate for Different drugs',
        labels={'Treated_with_drugs': 'Drug', 'Survived_1_year': 'Survival Rate'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    # Add text annotations with the survival rate percentages
    drug_fig.update_traces(texttemplate='%{y:.2%}', textposition='outside')

    # Display the drug survival bar plot
    st.plotly_chart(drug_fig)

def bmi_range():
    # Create BMI bins
    bins = [15, 18.5, 24.9, 30, 35, 40, 50]
    labels = ['Underweight', 'Normal weight', 'Overweight', 'Obesity I', 'Obesity II', 'Obesity III']
    df1['BMI Range'] = pd.cut(df1['Patient_Body_Mass_Index'], bins=bins, labels=labels)

    # Grouping by BMI range, Drugs, and Smoker Status
    grouped = df1.groupby(['Treated_with_drugs','BMI Range'])['Survived_1_year'].mean().reset_index()

    # Visualization using Plotly
    bmi_fig = px.scatter(
        grouped,
        x='Treated_with_drugs',
        y='Survived_1_year',
        color='BMI Range',
        labels={'Survived_1_year': 'Survival 1 Year', 'BMI Range': 'BMI Category'},
        title='Survival Rate by BMI Range, and Drugs',
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    bmi_fig.update_layout(legend_title='BMI Category')
    # Display the BMI Range scatter plot
    st.plotly_chart(bmi_fig)

    
def confusion_matrix_plot():
    global df1  # Access the global variable df1
    # Define categorical columns
    df1_categorical = ['Treated_with_drugs','Patient_Smoker', 'Patient_Rural_Urban', 'Patient_mental_condition']
    
    # Perform one-hot encoding using pandas' get_dummies function
    one_hot_encoded_df = pd.get_dummies(df1, columns=df1_categorical)
    
    # Removing the label column from the DataFrame
    X = one_hot_encoded_df.drop(columns=['Survived_1_year'])
    y = one_hot_encoded_df['Survived_1_year']
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    
    # Instantiate and train the model
    best_estimator = GradientBoostingClassifier(max_depth=4, n_estimators=150, learning_rate=0.1, random_state=45)
    best_estimator.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = best_estimator.predict(X_test)

    # Calculate the confusion matrix
    gradientboost_cm = confusion_matrix(y_test, y_pred)

    # Define class labels
    class_labels = ['Not Survived', 'Survived']

    # Create a Plotly heatmap for the confusion matrix with annotations
    cm_fig = go.Figure(data=go.Heatmap(
        z=gradientboost_cm,
        x=class_labels,
        y=class_labels,
        colorscale='greens',
        colorbar=dict(title='Count')
    ))

    # Add text annotations
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            cm_fig.add_annotation(
                text=str(gradientboost_cm[i, j]),
                x=class_labels[j],
                y=class_labels[i],
                showarrow=False,
                font=dict(color='white' if gradientboost_cm[i, j] > np.max(gradientboost_cm) / 2 else 'black')
            )

    # Customize the layout
    cm_fig.update_layout(
        title='Gradient Boosting Confusion Matrix',
        xaxis_title='Predicted Labels',
        yaxis_title='True Labels'
    )
# Display the confusion matrix heatmap
    st.plotly_chart(cm_fig)
    
    
def roc_curve_plot():
    global df1
    # Define categorical columns
    df1_categorical = ['Treated_with_drugs', 'Patient_Smoker', 'Patient_Rural_Urban', 'Patient_mental_condition']
    
    # Perform one-hot encoding using pandas' get_dummies function
    one_hot_encoded_df = pd.get_dummies(df1, columns=df1_categorical)
    
    # Removing the label column from the DataFrame
    X = one_hot_encoded_df.drop(columns=['Survived_1_year'])
    y = one_hot_encoded_df['Survived_1_year']
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    
    # Instantiate and train the model
    best_estimator = GradientBoostingClassifier(max_depth=4, n_estimators=150, learning_rate=0.1, random_state=45)
    best_estimator.fit(X_train, y_train)
    
    # Calculate ROC curve and AUC
    y_prob = best_estimator.predict_proba(X_test)[:, 1]  # Probability of the positive class
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Create ROC curve plot using Plotly
    roc_curve_fig = go.Figure()
    roc_curve_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC = %0.2f)' % roc_auc))
    roc_curve_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    roc_curve_fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        showlegend=True
    )

    # Display the ROC curve plot
    st.plotly_chart(roc_curve_fig)

# Define the Streamlit app


# Define the Streamlit app
def main():
    # Set up the sidebar and main content area
    st.sidebar.title('Navigation')
    options = ['Correlation Heatmap', 'Age Bin Survival', 'Drug Survival', 'BMI Range', 'Confusion Matrix', 'ROC Curve']
    choice = st.sidebar.radio('Select Visualization:', options)

    # Display selected visualization
    if choice == 'Correlation Heatmap':
        correlation_heatmap()
    elif choice == 'Age Bin Survival':
        age_bin_survival()
    elif choice == 'Drug Survival':
        drug_survival()
    elif choice == 'BMI Range':
        bmi_range()
    elif choice == 'Confusion Matrix':
        confusion_matrix_plot()
    elif choice == 'ROC Curve':
        roc_curve_plot()

if __name__ == '__main__':
    main()
