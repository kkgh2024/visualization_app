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

    # Customize the layout
    heatmap_fig.update_layout(
        title='Correlation Heatmap',
    )

 

    # Display the plot
    st.plotly_chart(heatmap_fig)

    # Display the description text
    additional_text_1 = """
    **Are our variables correlated?**  
    """
    additional_text_2 = """
    In general, the heatmap visualization of the correlation matrix facilitates straightforward interpretation of the\
    relationships among variables within the dataset. Incorporating text annotations further improves the clarity of\
    the heatmap by offering exact correlation coefficient values.
    """
    additional_text_3 = """
    **Interpretation**  
    """
    additional_text_4 = """
    The observed correlations are predominantly weak, suggesting that the variables in the dataset exhibit little linear \
    relationship with each other. Consequently, alternative methods like regression or machine\
    learning models may prove more suitable for exploring these relationships and predicting outcomes within this dataset.  
    """
    st.write(additional_text_1)
    st.write(additional_text_2)
    st.write(additional_text_3)
    st.write(additional_text_4)
    
def binary_response_distribution():
    # Calculate the percentage of each response category
    response_percentages = df1['Survived_1_year'].value_counts() / len(df1)

    # Create a bar plot using Plotly
    fig = px.bar(
        x=response_percentages.index,
        y=response_percentages.values,
        color=response_percentages.index,
        labels={'x': 'Survived_1_year', 'y': 'Percentage'},
        title='Distribution of Binary Response'
    )

    # Customize colors for the two response categories
    fig.update_traces(marker_color=['blue', 'red'])
    fig.update_traces(texttemplate='%{y:.2%}', textposition='outside')
    # Adjust the width of the bins by changing the bargap
    fig.update_layout(bargap=0.5)  # You can adjust this value

    # Display the plot
    st.plotly_chart(fig)

    additional_text_1 = """
    **Is the binary response variable 'Survived_1_year' balanced?** 
    """
    additional_text_2 = """
    The plot effectively communicates the distribution of the binary response variable 'Survived_1_year' in the dataset,\
    offering insights into the proportion of patients who survived after one year of treatment. \
    The utilization of colors, labels, and textual displays enhances the clarity and interpretability of the visualization.
    """
    additional_text_3 = """
    **interpretation**
    """
    additional_text_4 = """
    The "Survived_1_year" variable demonstrates a balanced distribution, suggesting that both outcomes\
    (survival and non-survival after one year of treatment) are evenly represented in the dataset. Specifically, \
    around 63.2% of patients survived after one year, \
    while approximately 36.8% did not survive. 
    """
    st.write(additional_text_1)
    st.write(additional_text_2)
    st.write(additional_text_3)
    st.write(additional_text_4)
    # Add interpretation text   

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
        # Add additional text
 #   additional_text = """
#    Generally, the survival rate tends to be higher for patients who are not smokers compared to those who are smokers within the same age group.
#    Among different age groups, younger patients (e.g., those aged 20-30 years) tend to have higher survival rates compared to older patients (e.g., those aged 60-70 years).
#    There is a significant difference in survival rates between smokers and non-smokers, particularly in older age groups. For example, in the age group 60-70, the survival rate for non-smokers is substantially higher than that for smokers.
#    """

#    st.write(additional_text)
     # Add additional text as three different points
    additional_text_1 = """
    **How does age and smoking status impact the survival rate?**
    """
    additional_text_2 = """
    The barplot effectively visualizes the survival rate for different age bins, stratified by smoker status. \
    It communicates how survival rates vary across age groups and smoker statuses 
    """
    additional_text_3 = """
    **Interpretation**
    """
    additional_text_4 = """
    Generally, the survival rate tends to be higher for non-smokers compared to smokers within the same age group.
    """
    additional_text_5 = """
    Among different age groups, younger patients (e.g., aged 20-30 years) typically exhibit higher survival rates\
    than older patients (e.g., aged 60-70 years).
    """
    additional_text_6 = """
    There is a significant disparity in survival rates between smokers and non-smokers, particularly among older age groups.\
    For instance, \
    in the 60-70 age group, non-smokers demonstrate substantially higher survival rates than smokers.
    """
    st.write(additional_text_1)
    st.write(additional_text_2)
    st.write(additional_text_3)
    st.write(additional_text_4)
    st.write(additional_text_5)
    st.write(additional_text_6)
    
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
          # Add additional text as three different points
    additional_text_1 = """
    **How do various drugs impact the treatment of smokers and non-smokers patients?**
    """
    additional_text_2 = """
    Overall, this plot effectively illustrates the impact of different drugs and smoking status on the survival rate.\
    It offers a visual comparison of survival rates across various drug treatments and smoker statuses,\
    enabling viewers to readily identify trends and patterns in the data. 
    """
    additional_text_3 = """
    **Interpretaion**
    """
    additional_text_4 = """
    a trend emerges where non-smokers generally demonstrate higher survival rates compared to smokers across various\
    drug treatments. Additionally, certain drug combinations exhibit higher survival rates compared to others,\
    irrespective of smoking status.
    """
    st.write(additional_text_1)
    st.write(additional_text_2)
    st.write(additional_text_3)
    st.write(additional_text_4)
def bmi_range_survival():
    # Create BMI bins
    bins = [15, 18.5, 24.9, 30, 35, 40, 50]
    labels = ['Underweight', 'Normal weight', 'Overweight', 'Obesity I', 'Obesity II', 'Obesity III']
    df1['BMI Range'] = pd.cut(df1['Patient_Body_Mass_Index'], bins=bins, labels=labels)

    # Group by BMI Range and Smoke, calculate mean SurvivalRate
    grouped = df1.groupby(['BMI Range', 'Patient_Smoker'])['Survived_1_year'].mean().reset_index()

    # Create a bar plot using Plotly
    fig = px.bar(
        grouped,
        x='BMI Range',
        y='Survived_1_year',
        color='Patient_Smoker',
        title='Survival Rate for Different BMI Range',
        labels={'BMI Range': 'BMI Range', 'Survived_1_year': 'Survival Rate'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    # Add text annotations with the survival rate percentages
    fig.update_traces(texttemplate='%{y:.2%}', textposition='outside')

    # Display the BMI Range survival bar plot
    st.plotly_chart(fig)
    additional_text_1 = """
    **How does BMI and smoking statue impact the survival rate?**
    """
    additional_text_2 = """
    This plot effectively illustrates the impact of BMI and smoking status on the survival rate. \
    It enables a visual comparison of survival rates across various BMI ranges and smoker statuses, \
    offering insights into potential correlations between these factors and survival outcomes.
    """
    additional_text_3 = """
    **Interpretation**
    """
    additional_text_4 = """
    Across all BMI ranges, non-smokers generally demonstrate higher survival rates compared to smokers.
    """
    additional_text_5 = """
    Within each BMI range, there is a consistent trend of decreasing survival rates for smokers compared to non-smokers.
    """
    additional_text_6 = """
    Overall, these results suggest that both BMI and smoking status impact the survival rate, with non-smokers and individuals with\
    lower BMI generally exhibiting higher survival rates.
    """
    st.write(additional_text_1)
    st.write(additional_text_2)
    st.write(additional_text_3)
    st.write(additional_text_4)
    st.write(additional_text_5)
    st.write(additional_text_6)
    # Add interpretation text
    
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
    additional_text_1 = """
    **Which are the most effective drugs for each BMI range?** 
    """
    additional_text_2 = """
    This plot facilitates exploration of the associations between different drug treatments, BMI categories,\
    and survival rates. It allows viewers to identify potential trends or patterns in survival outcomes based on drug\
    treatment and BMI range, \
    contributing to a deeper understanding of the dataset.
    """
    additional_text_3 = """
    **Interpretation** 
    """
    additional_text_4 = """
    The survival rates vary across different combinations of drug treatments and BMI ranges,\
    with some combinations showing higher survival rates. 
    """
    additional_text_5 = """
    For instance, patients treated with 'DX1 DX2 DX3 DX4' in the 'Normal weight' BMI range exhibit a mean survival rate\
    of 0.978723, indicating a high survival rate
    """
    additional_text_6 = """
     Conversely, certain combinations have lower survival rates, such as patients treated with 'DX6' in the 'Underweight' BMI range,\
     with a mean survival rate of 0.278008.
    """
    additional_text_7 = """
    Overall, the data provides insights into how different combinations of drug treatments and BMI ranges correlate\
    with patient survival rates.
    """
    
    st.write(additional_text_1)
    st.write(additional_text_2)
    st.write(additional_text_3)
    st.write(additional_text_4)
    st.write(additional_text_5)
    st.write(additional_text_6)
    st.write(additional_text_7)
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
    
    additional_text_1 = """
    **How do we evaluate the performance of our classification model?** 
    """
    additional_text_2 = """
    This design effectively illustrates the process of training a Gradient Boosting Classifier, \
    evaluating its performance using a confusion matrix,\
    and visualizing the results to gain insights into the model's predictive capabilities.
    """
    additional_text_3 = """
    **Interpretation** 
    """
    additional_text_4 = """
    The model correctly predicted 2524 instances of survival (True Positives) and 1157 instances of non-survival (True Negatives). However, it also made 553 incorrect predictions of survival (False Positives) \
    and 383 incorrect predictions of non-survival (False Negatives). 
    """
    additional_text_5 = """
    Overall, the model appears to have higher accuracy, 0.90,  in predicting survival (positive class) than non-survival (negative class).\
    However, there are still notable misclassifications,\
    particularly in the form of false positives and false negatives.
    """
    st.write(additional_text_1)
    st.write(additional_text_2)
    st.write(additional_text_3)
    st.write(additional_text_4)
    st.write(additional_text_5)
    
    
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
    
    additional_text_1 = """
    **How do we evaluate the performance of our classification model?** 
    """
    additional_text_2 = """
    This design and implementation enable clear visualization of the model's performance in discriminating between\
    positive and negative classes,\
    facilitating the assessment of its effectiveness in classification tasks.
    
    """
    additional_text_3 = """
    **Interpretation** 
    """
   
    additional_text_4 = """
    By analyzing the ROC curve generated, A model with high performance will have an ROC curve that hugs the
    upper left corner of the plot, indicating high true positive rate and low false positive rate across various threshold 
    values. The Area Under the ROC Curve (AUC) provides a single scalar value to summarize the overall performance of the model: the closer the AUC value is to 1, the better the model's \
    ability to distinguish between positive and negative classes across all possible threshold levels. 
 
    """
    
    st.write(additional_text_1)
    st.write(additional_text_2)
    st.write(additional_text_3)
    st.write(additional_text_4)
#   # Define the Streamlit app


# Define the Streamlit app
def main():
    # Set up the sidebar and main content area
    st.sidebar.title('Navigation')
    options = ['Correlation Heatmap', 'Binary Response Distribution','Age Bin Survival', 'Drug Survival','BMI Range Survival', 'BMI Range', 'Confusion Matrix', 'ROC Curve']
    choice = st.sidebar.radio('Select Visualization:', options)

    # Display selected visualization
    if choice == 'Correlation Heatmap':
        correlation_heatmap() 
        
    elif choice == 'Binary Response Distribution':
        binary_response_distribution()
        
    elif choice == 'Age Bin Survival':
        age_bin_survival()
    elif choice == 'Drug Survival':
        drug_survival()
    elif choice == 'BMI Range Survival':
        bmi_range_survival()    
    elif choice == 'BMI Range':
        bmi_range()
    elif choice == 'Confusion Matrix':
        confusion_matrix_plot()
    elif choice == 'ROC Curve':
        roc_curve_plot()

if __name__ == '__main__':
    main()
