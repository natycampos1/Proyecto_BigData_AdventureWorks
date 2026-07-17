import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("Product.csv", sep="\t")


df.columns = df.columns.str.strip()


df["Standard Cost"] = (
    df["Standard Cost"].astype(str).str.replace(r"[\$,]", "", regex=True).astype(float)
)


print(df.head())
print(df.info())


print(df["Standard Cost"].describe())


print(df["Category"].value_counts())


plt.hist(df["Standard Cost"], bins=10)


plt.title("Distribucion de Standard Cost")
plt.xlabel("Standard Cost")
plt.ylabel("Cantidad")
plt.show()



plt.boxplot(df["Standard Cost"])
plt.title("boxplot de Standard Cost")
plt.show()



# NOTA: En la plantilla del profe se cruzan dos columnas numéricas. 
# En tus datos podemos usar 'Standard Cost' e 'ProductKey' (o el índice) para el scatter plot:
plt.scatter(df["Standard Cost"], df.index)
plt.xlabel("Indice")
plt.ylabel("Standard Cost")
plt.show()



df["CostoDistribucion"] = pd.cut(
    df["Standard Cost"],
    bins=[0, 50, 300, 3000],  
    labels=["Bajo 50", "Medio 300", "Alto 3000"]
)

print(df["CostoDistribucion"].value_counts())


train, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Category"] 
)


train2, test2 = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["Category"]
)


print("train: ", train.shape)
print("test: ", test.shape)


print("Original")
print(df["Category"].value_counts(normalize=True))

print("\ntrain")
print(train["Category"].value_counts(normalize=True))

print("\ntrain2")
print(train2["Category"].value_counts(normalize=True))