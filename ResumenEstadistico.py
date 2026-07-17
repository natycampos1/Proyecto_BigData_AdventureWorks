import pandas as pd
import numpy as np

sales = pd.read_csv('Sales.csv', sep='\t')
product = pd.read_csv('Product.csv', sep='\t')
reseller = pd.read_csv('Reseller.csv', sep='\t')
salesperson = pd.read_csv('Salesperson.csv', sep='\t')
salesperson_region = pd.read_csv('SalespersonRegion.csv', sep='\t')
region = pd.read_csv('Region.csv', sep='\t')
targets = pd.read_csv('Targets.csv', sep='\t')

def clean_currency(val):
    if isinstance(val, str):
        val = val.replace('$', '').replace(',', '').strip()
        if '(' in val and ')' in val:
            val = '-' + val.replace('(', '').replace(')', '')
        return float(val)
    return val

# Clean money columns
sales['Unit Price'] = sales['Unit Price'].apply(clean_currency)
sales['Sales'] = sales['Sales'].apply(clean_currency)
sales['Cost'] = sales['Cost'].apply(clean_currency)

product['Standard Cost'] = product['Standard Cost'].apply(clean_currency)

targets['Target'] = targets['Target'].apply(clean_currency)

# Clean date columns
sales['OrderDate'] = pd.to_datetime(sales['OrderDate'])
targets['TargetMonth'] = pd.to_datetime(targets['TargetMonth'])

# Numerical summaries
sales_desc = sales[['Quantity', 'Unit Price', 'Sales', 'Cost']].describe().to_string()
product_desc = product[['Standard Cost']].describe().to_string()
targets_desc = targets[['Target']].describe().to_string()

# Categorical columns summaries
cat_summaries = {}

# Product categories and subcategories
cat_summaries['Product_Category'] = product['Category'].value_counts()
cat_summaries['Product_Color'] = product['Color'].value_counts(dropna=False)

# Reseller Business Type and top countries
cat_summaries['Reseller_BusinessType'] = reseller['Business Type'].value_counts()
cat_summaries['Reseller_Country'] = reseller['Country-Region'].value_counts()

# Region Groups
cat_summaries['Region_Group'] = region['Group'].value_counts()

# Salesperson titles
cat_summaries['Salesperson_Title'] = salesperson['Title'].value_counts()

# Output the findings
print("--- Venta de Producto ---")
print(sales_desc)
print("\n--- Costo del Producto ---")
print(product_desc)
print("\n--- Metas de Venta ---")
print(targets_desc)

print("\n--- Resumen de Categorias ---")
for k, v in cat_summaries.items():
    print(f"\n{k}:")
    print(v)