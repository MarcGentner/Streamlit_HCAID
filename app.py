import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split  # Import accuracy_score from sklearn.metrics
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import streamlit as st

import seaborn as sns

from sklearn.model_selection import train_test_split




# /-----------------------Import data -----------------------------\

df = pd.read_csv("model_data.csv")
data = pd.DataFrame(df)





# /-----------------------data cleaning -----------------------------\

# data = data.drop(['popularity','Unnamed: 18', 'Unnamed: 11', 'Cochrane systematic review', 'OTW', 'efficacy', 'alt name', 'main study source', 'Link to main individual study','Link to individual study','Link to individual study.1'], axis=1)

data['N positive studies / trials'] = pd.to_numeric(data['N positive studies / trials'], errors='coerce')
data['% positive studies/ trials'] = data['% positive studies/ trials'].str.rstrip('%').astype(float) / 100


# st.write("data",data)



# /----------------------- Heading -----------------------------\

# Create the app title and header
st.title("Supplement Recommendation App")
# st.subheader("Choose your input values to predict the best fitting supplement.")
st.write("Click the menu and select your preferences; the model will automatically select the top three best supplements based on research levels and popularity.")


# Load the trained model
# with open('./Model/model_clf.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)


target = 'supplement'
encode = ['claimed improved aspect of fitness', 'fitness category', 'sport or exercise type tested']

one = 'claimed improved aspect of fitness'
two = 'fitness category'
tre = 'sport or exercise type tested'

data = pd.get_dummies(data, columns=[one], prefix=one)
data = pd.get_dummies(data, columns=[two], prefix=two)
data = pd.get_dummies(data, columns=[tre], prefix=tre)

# st.write("encoded: ", data)



# target_mapper = {}
target_mapper = {supplement: idx for idx, supplement in enumerate(data['supplement'].unique())}

def target_encode(val):
    return target_mapper[val]


data['supplement'] = data['supplement'].apply(target_encode)

# df = df.drop(['claimed improved aspect of fitness', 'fitness category'])

X = data.drop('supplement', axis = 1)
y = data['supplement']
# st.write("X",X)
# st.write("Y",y)

# train random forrest: 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# trainen: 
# predict (output: [int] 0 - 46)
# model.fit(X, y)
model.fit(X_train, y_train)








# user input features 


# st.write("gettting unique data: ", data)
uniq1 = df['claimed improved aspect of fitness'].unique()
uniq2 = df['fitness category'].unique()
uniq3 = df['sport or exercise type tested'].unique()
# 
unique_values = len(uniq1), len(uniq2), len(uniq3)
# st.write(unique_values)


def user_input_features(data):
    selected_aspects= st.sidebar.selectbox('Select Claimed Improved Aspects of Fitness', uniq1)
    selected_categories = st.sidebar.selectbox('Select Fitness Categories', uniq2)
    selected_types = st.sidebar.selectbox('Select Sport or Exercise Types', uniq3)
        
       
    input_data = {
        'evidence level': st.sidebar.selectbox('Select Evidence Level', df['evidence level'].unique()),
        'popularity': st.sidebar.slider('Select Popularity', int(df['popularity'].min()), int(df['popularity'].max()), int(df['popularity'].mean())),
        'number of studies examined': st.sidebar.slider('Select Number of Studies Examined', int(df['number of studies examined'].min()), int(df['number of studies examined'].max()), int(df['number of studies examined'].mean()))
    }

    input_data = data.iloc[0].drop("supplement").to_dict()
    # input_data.pop('% positive studies/ trials', None)
    # st.write(len(input_data))

    encoded_cols = [col for col in data.columns if col.startswith('claimed improved aspect of fitness_') or col.startswith('fitness category_') or col.startswith('sport or exercise type tested_')]
    # st.write("encoded: ", encoded_cols)
    for col in encoded_cols:
        input_data[col] = 0  # Set all encoded columns to 0 by default

    # st.write(len(input_data))




    # Prepend the selected aspects, categories, and types to the column names and set them to 1
    for aspect in selected_aspects:
        column_name = f'Claimed improved aspect of fitness_{aspect}'
        input_data[column_name] = 1
        # data = data.drop(column_name, axis=1)
        del input_data[column_name]

    # st.write("Aspect", len(input_data))
    # st.write("Aspect", input_data)



    for category in selected_categories:
        column_name = f'fitness category_{category}'
        input_data[column_name] = 1
        del input_data[column_name]
        # data = data.drop(column_name, axis=1)
    # st.write("categorie", len(input_data))
    # st.write("categorie", input_data)

    for typ in selected_types:
        column_name = f'sport or exercise type tested_{typ}'
        input_data[column_name] = 1
        del input_data[column_name]
    # st.write("categorie", len(input_data))
    # st.write("categorie", input_data)
        # data = data.drop(column_name, axis=1)

    input_df = pd.DataFrame(input_data, index=[0])
    return input_df

input_df = user_input_features(data)
# st.write("input length", len(input_df))
# st.write("input: ", input_df)
# st.write("input data",input_df)

# st.write("Number of columns in input_df:", input_df.shape[1])
# st.write("Number of columns in X:", X.shape[1])


# Check if at least one input feature is selected
if input_df.empty:
    st.warning('Please select at least one input feature to make a prediction.')
else:
    # st.write("Input data:")
    # st.write(input_df)
    st.subheader("Your prediction")

    if st.button("Predict Supplement"):
     
        # Make a prediction using the encoded input data
        prediction = model.predict(input_df)[0]

        # Get the top 3 supplements
        top_3_supplements = model.predict_proba(input_df)[0].argsort()[-3:]
        # Fit the model to the test data
        y_pred = model.predict(X_test)

    # Calculate accuracy on the test data
        accuracy = accuracy_score(y_test, y_pred)

    # Print accuracy

    # Plot a confusion matrix to visualize the performance of the model


        # Show the accuracy of the top 3 supplements predicted
        st.write('Top 3 supplements and their probabilities:')
        for supplement in top_3_supplements:
            predicted_supplement = next(k for k, v in target_mapper.items() if v == supplement)
            probability = model.predict_proba(input_df)[0][supplement]
            st.write(f'- {supplement} {predicted_supplement}: {probability:.2f}')
    
        st.write(f'Model Accuracy: {accuracy * 100:.2f}%')



        st.subheader("About this prediction:")

            
        # Display a classification report
        report = classification_report(y_test, y_pred)
        st.text("Classification Report:")
        st.text(report)

        from sklearn.tree import plot_tree
        from sklearn.tree import export_text


        # Plot an individual tree from the Random Forest
        st.subheader("Random Forest Tree Visualization")
        if st.button("Plot the Tree"):
            from sklearn.tree import plot_tree
            tree_to_plot = model.estimators_[0]  # You can select a different tree if needed
            plt.figure(figsize=(15, 10))
            plot_tree(tree_to_plot, feature_names=list(X.columns), filled=True, class_names=[str(i) for i in range(len(target_mapper))])
            st.pyplot(plt)
        # predicted_supplement = next(k for k, v in target_mapper.items() if v == prediction)

        # st.write(f'Predicted Supplement: {predicted_supplement}')

    





