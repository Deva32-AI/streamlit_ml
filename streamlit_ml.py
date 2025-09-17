import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("📊 Data Science using ML Algorithms - Hybrid Model")

learning_type = st.sidebar.radio("Select Learning Type", ["Supervised", "Unsupervised"])
if learning_type == "Supervised":
    algorithm = st.sidebar.selectbox("Choose Supervised Algorithm", ["Logistic Regression", "KNN"])
else:
    algorithm = st.sidebar.selectbox("Choose Unsupervised Algorithm", ["KMeans", "PCA"])

upload_file = st.file_uploader("📂 Upload Excel/CSV file", type=["xlsx", "xls", "csv"])

if upload_file is not None:
    if upload_file.name.endswith(".csv"):
        df = pd.read_csv(upload_file)
    else:
        df = pd.read_excel(upload_file)

    st.success("✅ File uploaded successfully!")
    st.subheader("📑 Dataset Preview")
    st.dataframe(df.head())

    if learning_type == "Supervised":
        st.subheader("⚡ Supervised Learning")
        target_column = st.selectbox("🎯 Select the Target Column", df.columns)
        x = df.drop(columns=[target_column])
        y = df[target_column]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        if st.button("🚀 Train Model"):
            if algorithm == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif algorithm == "KNN":
                model = KNeighborsClassifier()

            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            cm = confusion_matrix(y_test, predictions)

            st.success("✅ Model trained successfully!")
            st.write(f"**Accuracy:** {accuracy:.2f}")

            st.subheader("📊 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            st.subheader("📋 Classification Report")
            st.text(classification_report(y_test, predictions))

    else:
        st.subheader("🔍 Unsupervised Learning")
        features = st.multiselect("📌 Select Features for Unsupervised Learning", df.columns, default=df.columns.tolist())
        X = df[features]

        if st.button("🚀 Run Unsupervised Algorithm"):
            if algorithm == "KMeans":
                model = KMeans(n_clusters=3, random_state=42)
                clusters = model.fit_predict(X)
                df["Cluster"] = clusters
                st.success("✅ KMeans clustering completed!")
                st.dataframe(df.head())

                st.subheader("📊 Cluster Visualization")
                fig, ax = plt.subplots()
                if X.shape[1] >= 2:
                    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=df["Cluster"], palette="Set2", ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("Need at least 2 features for cluster visualization.")

            elif algorithm == "PCA":
                model = PCA(n_components=2)
                components = model.fit_transform(X)
                df["PCA1"] = components[:, 0]
                df["PCA2"] = components[:, 1]
                st.success("✅ PCA dimensionality reduction completed!")
                st.dataframe(df.head())

                st.subheader("📊 PCA Visualization")
                fig, ax = plt.subplots()
                sns.scatterplot(x="PCA1", y="PCA2", data=df, palette="Set1", ax=ax)
                st.pyplot(fig)
else:
    st.warning("⚠️ Please upload an Excel or CSV file to continue.")
