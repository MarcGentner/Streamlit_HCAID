import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split  # Import accuracy_score from sklearn.metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



data = pd.read_csv("cleaned_data.csv")











unique_values = data['Claimed improved aspect of fitness'].unique()
unique_values1 = data['sport or exercise type tested'].unique()
unique_values2 = data['fitness category'].unique()

new_data = pd.DataFrame()

# Create a multiselect input for user selection

with st.sidebar: 
    def user_input_features():
        st.header("User Input")

        sci = st.multiselect("Select Claimed Improved Aspect of Fitness:", unique_values)
        set = st.multiselect("Sport or exercise type tested:", unique_values1)
        fc = st.multiselect("Fitness category:", unique_values2)

        data = {
            'sci':sci,
            'set':set,
            'fc':fc,
        }
        features = pd.DataFrame(data, index=[0])

        return features
    input_df = user_input_features() 

# # Add columns to the new data frame for the selected values
# new_data["Claimed Improved Aspect of Fitness"] = sci
# new_data["Sport or exercise type tested"] = set
# new_data["Fitness category"] = fc


# st.write(new_data)


st.write("userinput data: ", data)


from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'supplement' column
encoded_supplements = label_encoder.fit_transform(data['supplement'])

# Create a mapping dictionary
supplement_mapping = {label: supplement for label, supplement in zip(encoded_supplements, data['supplement'])}

# Replace the original 'supplement' column with the encoded values
data['supplement'] = encoded_supplements





def clean_data(data):
    """Cleans the data by converting the `N positive studies / trials` column to numeric and the `% positive studies/ trials` column to values between 0 and 1."""

    data = pd.DataFrame(data)
    data['N positive studies / trials'] = pd.to_numeric(data['N positive studies / trials'], errors='coerce')
    data['% positive studies/ trials'] = data['% positive studies/ trials'].str.rstrip('%').astype(float) / 100

    return data


    # Clean the data
data = clean_data(data)







# Prepare the data
X = data.drop(['supplement', 'Claimed improved aspect of fitness', 'fitness category', 'sport or exercise type tested', 'Link to individual study.1', 'alt name', 'Link to individual study', 'Link to main individual study', 'main study source'], axis=1)
y = data["supplement"]

# st.write("X:",X)
# st.write("y:",y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# st.write("X_train:",X_train)
# st.write("y_train:",y_train)
# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)





# Check if user has selected any aspects
# if sci or set or fc:
    # Filter data based on user's selection
filtered_data = data[
    (data['Claimed improved aspect of fitness'].isin(sci)) &
    (data['sport or exercise type tested'].isin(set)) &
    (data['fitness category'].isin(fc))
]
encoded_data = pd.get_dummies(filtered_data, columns=['Claimed improved aspect of fitness', 'sport or exercise type tested', 'fitness category'])
X_input = encoded_data.drop(['supplement'], axis=1)

y_pred_proba = model.predict_proba(X_input)

supplement_names = label_encoder.inverse_transform(model.classes_)

# Create a DataFrame to store the supplements and their prediction probabilities
supplement_predictions = pd.DataFrame({
    "Supplement": supplement_names,
    "Prediction Probability": y_pred_proba[0]
})

# Sort the DataFrame by prediction probability in descending order
supplement_predictions = supplement_predictions.sort_values(by="Prediction Probability", ascending=False)

# Display the top 3 supplements with the highest prediction probabilities
st.header("Top 3 Predicted Supplements")
st.table(supplement_predictions.head(3))


    # # # Clean the data
    # # filtered_data = clean_data(filtered_data)

    # # # Prepare the input data for prediction
    # # X_input = filtered_data.drop(['supplement', 'Claimed improved aspect of fitness', 'fitness category', 'sport or exercise type tested', 'Link to individual study.1', 'alt name', 'Link to individual study', 'Link to main individual study', 'main study source'], axis=1)

    # # # Make predictions
    # # y_pred_proba = model.predict_proba(X_input)

    # # # Get the supplement names
    # # supplement_names = label_encoder.inverse_transform(model.classes_)

    # # # Create a DataFrame to store the supplements and their prediction probabilities
    # # supplement_predictions = pd.DataFrame({
    # #     "Supplement": supplement_names,
    # #     "Prediction Probability": y_pred_proba[0]
    # # })

    # # Sort the DataFrame by prediction probability in descending order
    # supplement_predictions = supplement_predictions.sort_values(by="Prediction Probability", ascending=False)

    # # Display the top 3 supplements with the highest prediction probabilities
    # st.header("Top 3 Predicted Supplements")
    # st.table(supplement_predictions.head(3))













# Convert the encoded labels in 'y_pred' and 'y_test' back to supplement names
y_pred_names = [supplement_mapping[label] for label in y_pred]
y_test_names = [supplement_mapping[label] for label in y_test]


# Display metrics with supplement names
st.write("Accuracy Score:", accuracy_score(y_test_names, y_pred_names))
st.write("Classification Report:\n", classification_report(y_test_names, y_pred_names))

# Display metrics
st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
st.write("Classification Report:\n", classification_report(y_test, y_pred))

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot()





































# Load the trained model
# model = pickle.load(open('trained_model.pkl', 'rb'))


# def user_input_features():
#     one = st.sidebar.multiselect('Claimed improved aspect of fitness', ('','','','',''))
#     tre =st.sidebar.multiselect(data['sport or exercise type tested'], ('','','','',''))
#     two =st.sidebar.multiselect('fitness category', ('','','','',''))

#     data = {
#         'one': one, 
#         'two': two, 
#         'tre': tre, 
#     }
#     features = pd.DataFrame(data, index=[0])
#     return features


# data = user_input_features()








