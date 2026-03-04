import joblib

loaded_model = joblib.load(r"C:\Task final LA\ml-predictia-categoriei-produsului-pe-baza-titlului\data\final_product_category_model.pkl")

prediction = loaded_model.predict(["bosch serie 4 kgv39vl31g"]) #"olympus e m10 mark iii geh use silber", "kenwood k20mss15 solo", "bosch wap28390gb 8kg 1400 spin", "bosch serie 4 kgv39vl31g", "smeg sbs8004po"
print(prediction)