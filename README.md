# ml-predictia-categoriei-produsului-pe-baza-titlului
Scopul acestei sarcini este să dezvolți un model de învățare automată care să sugereze automat categoria potrivită pentru fiecare produs nou, pe baza titlului său.
### INSTRUCTIUNI DE UTILIZARE

1. Se incarca datele si se formateaza cu ajutorul: **products_cleaned.py**;
2. Se foloseste CSV-ul rezultat in urma formatarii pentru a se testa mai multe modele. In urma rularii **test_train_model.py** se identifica modelul cel mai eficent, in acest caz LinearSVC;
3. Se foloseste modelul ales pentru a fi antrenat pe intregul set de date **final_train_model.py** unde se foloseste feature engineering pentru a determina reguli mai precise de identificare a categoriilor produselor;
4. Se face testarea prin **predict_category.py** unde se pune un esantion de produce pentru care se verifica predictia.

In final se poate spune ca fara feature engineering modelul nu ar fi identificat corect "bosch serie 4 kgv39vl31g", "smeg sbs8004po". 
