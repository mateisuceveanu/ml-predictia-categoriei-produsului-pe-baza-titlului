from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Task final LA\ml-predictia-categoriei-produsului-pe-baza-titlului\data\IMLP4_TASK_03-products_cleaned.csv")
X = df["product_title"]
y = df["category_label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": LinearSVC()
}

# Train, predict, and evaluate
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    print(f"\n{name} - Classification Report:")
    print(classification_report(y_test, y_pred))

#Conclusion
# În urma comparării mai multor algoritmi de clasificare (Logistic Regression, Naive Bayes, Decision Tree și Support Vector Machine), 
# modelul Support Vector Machine (LinearSVC) a obținut cele mai bune rezultate din punct de vedere al performanței generale.
# Acesta a înregistrat cea mai mare acuratețe (97%) și cel mai ridicat scor macro F1 (0.97), 
# demonstrând o capacitate foarte bună de generalizare și un echilibru excelent între precision și recall pentru toate categoriile de produse.
# Performanța ridicată este justificată și din punct de vedere teoretic: SVM este cunoscut pentru eficiența sa în probleme de clasificare text,
# mai ales atunci când datele sunt reprezentate prin vectori TF-IDF, care generează spații de dimensiuni mari și sparse.

# Comparativ cu:
# Logistic Regression, care a avut o performanță apropiată (96%), dar ușor inferioară,
# Naive Bayes, care a prezentat scăderi semnificative pe anumite clase,
# Decision Tree, care a avut rezultate mai puțin stabile,
# Support Vector Machine a oferit cel mai bun compromis între precizie, robustețe și consistență a rezultatelor.