import pandas as pd

df= pd.read_csv(r"C:\Task final LA\ml-predictia-categoriei-produsului-pe-baza-titlului\data\IMLP4_TASK_03-products.csv")

#Format column titles
df.columns = (
    df.columns
    .str.replace("_", " ")
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
print(df.columns)

#print number of rows and columns
print("Number of rows and columns:", df.shape)

#print the nuber of null per column
print("Missing values pe column:")
print(df.isna().sum())

#print unique val of sentiment column
print("Unique values:")
print(df['category_label'].unique())

#uniform format usefull columns
df["category_label"] = df["category_label"].str.lower().str.strip()
df["product_title"] = df["product_title"].str.lower().str.strip()
print(df["category_label"].unique())

#Drop na val
df = df.dropna()
print(df["category_label"].unique())

#normalize lables
df["category_label"] = df["category_label"].replace({
    "cpu": "cpus",
    "fridge": "fridges",
    "mobile phone": "mobile phones"
})

#save cleaned datasheet
df.to_csv(r"C:\Task final LA\ml-predictia-categoriei-produsului-pe-baza-titlului\data\IMLP4_TASK_03-products_cleaned.csv", index=False)