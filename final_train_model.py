from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Task final LA\ml-predictia-categoriei-produsului-pe-baza-titlului\data\IMLP4_TASK_03-products_cleaned.csv")

X = df["product_title"]
y = df["category_label"]

# Pipeline final (TF-IDF + SVM)
final_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LinearSVC())
])

# Antrenare pe tot dataset-ul
final_model.fit(X, y)

#Salvare model pentru utilizare ulterioara
joblib.dump(final_model, r"C:\Task final LA\ml-predictia-categoriei-produsului-pe-baza-titlului\data\final_product_category_model.pkl")