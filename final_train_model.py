from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd

# 1. Încărcare date
df = pd.read_csv(r"C:\Task final LA\ml-predictia-categoriei-produsului-pe-baza-titlului\data\IMLP4_TASK_03-products_cleaned.csv")

# Curățare minimă: transformăm în lowercase (Tfidf face asta implicit, dar e bine să fim siguri)
X = df["product_title"].str.lower()
y = df["category_label"]

# 2. Pipeline Optimizat
final_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),  # Recunoaște și combinații de două cuvinte (ex: "bosch serie")
        stop_words='english', # Elimină cuvintele de legătură
        max_df=0.9,           # Ignoră termenii care apar în prea multe documente
        min_df=2              # Ignoră termenii care apar doar o dată (posibile typo-uri)
    )),
    ("classifier", LinearSVC(
        class_weight='balanced', # Foarte important dacă ai mai multe mașini de spălat decât frigidere
        C=1.0, 
        max_iter=2000
    ))
])

# 3. Antrenare
final_model.fit(X, y)

# 4. Salvare
joblib.dump(final_model, r"C:\Task final LA\ml-predictia-categoriei-produsului-pe-baza-titlului\data\final_product_category_model.pkl")