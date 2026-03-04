import joblib
import pandas as pd
import re

# 1. TREBUIE să redefinești funcția aici, exact cum era în scriptul de antrenare
# Altfel joblib nu știe cum să proceseze datele!
def extract_smart_features(df):
    s = df.iloc[:, 0].astype(str).str.lower()
    features = pd.DataFrame(index=df.index)
    features['is_fridge_code'] = s.apply(lambda x: 1 if re.search(r'\bkg[vna]\d+', x) else 0)
    features['is_sbs'] = s.apply(lambda x: 1 if 'sbs' in x or 'side by side' in x else 0)
    features['has_litres'] = s.apply(lambda x: 1 if re.search(r'\d+\s?(l|ltr|litre|litri)\b', x) else 0)
    features['has_rpm'] = s.apply(lambda x: 1 if re.search(r'\d{3,}\s?rpm', x) else 0)
    features['has_kg_load'] = s.apply(lambda x: 1 if re.search(r'\b\d{1,2}\s?kg\b', x) else 0)
    return features

# 2. Acum poți încărca modelul fără eroare
model_path = r"C:\Task final LA\ml-predictia-categoriei-produsului-pe-baza-titlului\data\final_model_v4.pkl"
loaded_model = joblib.load(model_path)

# 3. Datele de test
test_list = [
    "iphone 7 32gb gold,4,3,Apple iPhone 7 32GB",
    "olympus e m10 mark iii geh use silber", 
    "kenwood k20mss15 solo", 
    "bosch wap28390gb 8kg 1400 spin", 
    "bosch serie 4 kgv39vl31g", 
    "smeg sbs8004po"
]

# 4. Transformare în DataFrame
df_test = pd.DataFrame(test_list, columns=['product_title'])

# 5. Predicție
predictions = loaded_model.predict(df_test)

for text, pred in zip(test_list, predictions):
    print(f"Produs: {text} --> Categorie: {pred}\n")