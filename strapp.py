import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("train.csv")

df = df.drop(df.select_dtypes("object").columns, axis=1)
df = df.fillna(df.mean())

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

def get_model_score(model):
    if model == models[0]:
        m = LogisticRegression()
    if model == models[1]:
        m = KNeighborsClassifier()
    if model == models[2]:
        m = RandomForestClassifier()
    
    m.fit(X_train, y_train)

    return m.score(X_test, y_test)

pages = ["Presentation", "Data Viz", "Model Demo"]

st.sidebar.title("Navigation")

page = st.sidebar.radio("Choose a page:", pages)

if page == pages[0]:

    st.title("Titanic")
    st.header("a machine learning project")
    st.subheader("by the DataScientest team")

    st.image("image.jpg")

    st.markdown("This is the [dataset](https://www.kaggle.com/datasets/brendan45774/test-file) we will use:")

    st.dataframe(df.head())

if page == pages[1]:

    disp = st.checkbox("display viz:")

    if disp:

        fig = plt.figure()

        sns.countplot(x="Survived", data=df)

        st.pyplot(fig)

if page == pages[2]:

    models = ["logreg", "knn", "Random Forest"]

    model = st.selectbox("Choose a model to train:", models)

    st.write("The score of the {} is: ".format(model), get_model_score(model))
