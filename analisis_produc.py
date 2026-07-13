import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Cargamos tu archivo con sep="\t" para que Pandas reconozca las columnas de AdventureWorks
df = pd.read_csv("Product.csv", sep="\t")

# Limpiamos los nombres de las columnas (quita espacios invisibles rebeldes)
df.columns = df.columns.str.strip()

# Limpiar Standard Cost: quitar "$" y "," y convertir a número (necesario en tus datos)
df["Standard Cost"] = (
    df["Standard Cost"].astype(str).str.replace(r"[\$,]", "", regex=True).astype(float)
)


print(df.head())
print(df.info())


print(df["Standard Cost"].describe())


print(df["Category"].value_counts())


# plt.hist(df["Standard Cost"], bins=10)


# plt.title("Distribucion de Standard Cost")
# plt.xlabel("Standard Cost")
# plt.ylabel("Cantidad")
# plt.show()



# plt.boxplot(df["Standard Cost"])
# plt.title("boxplot de Standard Cost")
# plt.show()



# NOTA: En la plantilla del profe se cruzan dos columnas numéricas. 
# En tus datos podemos usar 'Standard Cost' e 'ProductKey' (o el índice) para el scatter plot:
# plt.scatter(df["Standard Cost"], df.index)
# plt.xlabel("Indice")
# plt.ylabel("Standard Cost")
# plt.show()



df["CostoDistribucion"] = pd.cut(
    df["Standard Cost"],
    bins=[0, 50, 300, 3000],  # Ajustamos los rangos a los precios de tus productos
    labels=["Bajo 50", "Medio 300", "Alto 3000"]
)

print(df["CostoDistribucion"].value_counts())


train, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Category"]  # Estratificamos por tu columna de categorías
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