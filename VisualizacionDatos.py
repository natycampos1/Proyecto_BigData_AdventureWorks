import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar el estilo de los gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# 1. Cargar datos
sales = pd.read_csv('Sales.csv', sep='\t')
product = pd.read_csv('Product.csv', sep='\t')
reseller = pd.read_csv('Reseller.csv', sep='\t')
targets = pd.read_csv('Targets.csv', sep='\t')

def clean_currency(val):
    if isinstance(val, str):
        val = val.replace('$', '').replace(',', '').strip()
        if '(' in val and ')' in val:
            val = '-' + val.replace('(', '').replace(')', '')
        return float(val)
    return val

sales['Sales'] = sales['Sales'].apply(clean_currency)
sales['Cost'] = sales['Cost'].apply(clean_currency)
sales['Unit Price'] = sales['Unit Price'].apply(clean_currency)
product['Standard Cost'] = product['Standard Cost'].apply(clean_currency)
targets['Target'] = targets['Target'].apply(clean_currency)

# Crear directorio para guardar las imágenes si no existe
os.makedirs('plots', exist_ok=True)

# --- Gráfico 1: Distribución de Ventas (Histograma con KDE) ---
plt.figure()
# Filtramos valores extremos muy altos para apreciar mejor el cuerpo de la distribución
sales_filtered = sales[sales['Sales'] < 5000]
sns.histplot(sales_filtered['Sales'], bins=50, kde=True, color='royalblue')
plt.title('Distribución del Monto de Ventas por Transacción (Filtro < $5,000)')
plt.xlabel('Monto de Ventas ($)')
plt.ylabel('Frecuencia')
plt.tight_layout()
plot1_path = 'plots/distribucion_ventas.png'
plt.savefig(plot1_path)
plt.close()

# --- Gráfico 2: Ventas por Categoría de Producto ---
# Unimos Sales con Product para obtener la categoría
sales_product = pd.merge(sales, product, on='ProductKey', how='left')
sales_by_cat = sales_product.groupby('Category')['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False)

plt.figure()
sns.barplot(data=sales_by_cat, x='Sales', y='Category', palette='viridis')
plt.title('Ventas Totales por Categoría de Producto')
plt.xlabel('Ventas Totales ($)')
plt.ylabel('Categoría')
# Formatear el eje X para que sea legible en millones
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}M".format(int(x/1e6)) if x >= 1e6 else "{:,}".format(int(x))))
plt.tight_layout()
plot2_path = 'plots/ventas_por_categoria.png'
plt.savefig(plot2_path)
plt.close()

# --- Gráfico 3: Tipo de Negocio de los Resellers (Distribución) ---
plt.figure()
sns.countplot(data=reseller, x='Business Type', palette='Set2')
plt.title('Distribución de Distribuidores (Resellers) por Tipo de Negocio')
plt.xlabel('Tipo de Negocio')
plt.ylabel('Cantidad de Resellers')
plt.tight_layout()
plot3_path = 'plots/distribucion_resellers.png'
plt.savefig(plot3_path)
plt.close()

print("Plots saved successfully:")
print(plot1_path)
print(plot2_path)
print(plot3_path)