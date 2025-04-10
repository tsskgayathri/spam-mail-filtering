import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Page Title
st.set_page_config(page_title="Spam Mail Classifier", layout="centered")
st.title("📧 Spam Mail Classifier")
st.write("Enter a message to check if it's spam or not.")

# Load and train model
@st.cache_data
def load_model():
    data = pd.read_csv("mail_data.csv")
    data = data.where(pd.notnull(data), '')

    data.loc[data['Category'] == 'spam', 'Category'] = 0
    data.loc[data['Category'] == 'ham', 'Category'] = 1

    X = data['Message'].astype(str)
    Y = data['Category']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    encoder = LabelEncoder()
    Y_train = encoder.fit_transform(Y_train)
    Y_test = encoder.transform(Y_test)

    model = LogisticRegression()
    model.fit(X_train_features, Y_train)

    return model, vectorizer, X_test_features, Y_test

# Load the trained components
model, vectorizer, X_test_features, Y_test = load_model()

# Input field
input_message = st.text_area("✉️ Message Text", height=150)

# Predict button
if st.button("Predict"):
    if input_message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        input_data = vectorizer.transform([input_message])
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("✅ This is a **Ham** message.")
        else:
            st.error("🚨 This is a **Spam** message.")

# Accuracy (Optional)
if st.checkbox("Show model test accuracy"):
    accuracy = accuracy_score(Y_test, model.predict(X_test_features))
    st.write(f"**Test Accuracy:** {accuracy:.2%}")
