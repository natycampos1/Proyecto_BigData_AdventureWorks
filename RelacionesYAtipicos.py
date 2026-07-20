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


def clean_currency(val):
    if isinstance(val, str):
        val = val.replace('$', '').replace(',', '').strip()
        if '(' in val and ')' in val:
            val = '-' + val.replace('(', '').replace(')', '')
        return float(val)
    return val


sales['Unit Price'] = sales['Unit Price'].apply(clean_currency)
sales['Sales'] = sales['Sales'].apply(clean_currency)
sales['Cost'] = sales['Cost'].apply(clean_currency)
product['Standard Cost'] = product['Standard Cost'].apply(clean_currency)

# Cruzamos Sales con Product para tener Standard Cost, Category y Subcategory
# junto a las métricas de venta de cada transacción.
df = pd.merge(sales, product, on='ProductKey', how='left')

os.makedirs('plots', exist_ok=True)

num_cols = ['Quantity', 'Unit Price', 'Sales', 'Cost', 'Standard Cost']

# ---------------------------------------------------------------------------
# IDENTIFICACIÓN DE RELACIONES
# ---------------------------------------------------------------------------

# Matriz de correlación entre variables numéricas
corr = df[num_cols].corr()
print("--- Matriz de correlación ---")
print(corr.to_string())

plt.figure()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title('Correlación entre variables numéricas')
plt.tight_layout()
plt.savefig('plots/correlacion_variables.png')
plt.close()

# Relación Quantity vs Sales
plt.figure()
sns.scatterplot(data=df, x='Quantity', y='Sales', alpha=0.4)
plt.title('Relación entre Cantidad Vendida y Monto de Venta')
plt.xlabel('Quantity')
plt.ylabel('Sales ($)')
plt.tight_layout()
plt.savefig('plots/relacion_quantity_sales.png')
plt.close()

# Relación Unit Price vs Cost
plt.figure()
sns.scatterplot(data=df, x='Unit Price', y='Cost', alpha=0.4)
plt.title('Relación entre Precio Unitario y Costo')
plt.xlabel('Unit Price ($)')
plt.ylabel('Cost ($)')
plt.tight_layout()
plt.savefig('plots/relacion_unitprice_cost.png')
plt.close()

# Relación Standard Cost (producto) vs Unit Price (venta real)
plt.figure()
sns.scatterplot(data=df, x='Standard Cost', y='Unit Price', alpha=0.4)
plt.title('Relación entre Costo Estándar del Producto y Precio de Venta')
plt.xlabel('Standard Cost ($)')
plt.ylabel('Unit Price ($)')
plt.tight_layout()
plt.savefig('plots/relacion_standardcost_unitprice.png')
plt.close()

# ---------------------------------------------------------------------------
# IDENTIFICACIÓN DE VALORES ATÍPICOS (método IQR)
# ---------------------------------------------------------------------------

def resumen_outliers_iqr(data, columna):
    q1 = data[columna].quantile(0.25)
    q3 = data[columna].quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    atipicos = data[(data[columna] < limite_inferior) | (data[columna] > limite_superior)]

    return {
        'columna': columna,
        'Q1': q1,
        'Q3': q3,
        'IQR': iqr,
        'limite_inferior': limite_inferior,
        'limite_superior': limite_superior,
        'cantidad_atipicos': len(atipicos),
        'porcentaje_atipicos': round(len(atipicos) / len(data) * 100, 2),
    }


print("\n--- Resumen de valores atípicos (IQR) ---")
for col in num_cols:
    resumen = resumen_outliers_iqr(df, col)
    print(f"\n{col}:")
    for k, v in resumen.items():
        if k != 'columna':
            print(f"  {k}: {v}")

# Boxplots para visualizar los valores atípicos de cada variable
fig, axes = plt.subplots(1, len(num_cols), figsize=(18, 5))
for ax, col in zip(axes, num_cols):
    sns.boxplot(y=df[col], ax=ax, color='skyblue')
    ax.set_title(col)
plt.suptitle('Valores atípicos por variable (boxplot)')
plt.tight_layout()
plt.savefig('plots/atipicos_boxplots.png')
plt.close()

print("\nGráficos guardados en la carpeta 'plots/':")
print("  plots/correlacion_variables.png")
print("  plots/relacion_quantity_sales.png")
print("  plots/relacion_unitprice_cost.png")
print("  plots/relacion_standardcost_unitprice.png")
print("  plots/atipicos_boxplots.png")
