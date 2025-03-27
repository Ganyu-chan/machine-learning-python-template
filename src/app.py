from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
 
datos = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")
datos.head()
datos.to_csv("/workspaces/machine-learning-python-template/data/raw/AB_NYC_2019.csv", index = False)

Columnas = datos.shape[1]
Filas = datos.shape[0]
print(f"El dataframe contiene {Columnas} Columnas y {Filas} Filas\n")

datos.info()
# Contar NaN por columna
nulos_por_columna = datos.isnull().sum()

# Encontrar el valor máximo de NaN
max_nulos = nulos_por_columna.max()

# Filtrar todas las columnas que tienen ese máximo de NaN
columnas_mas_nulos = nulos_por_columna[nulos_por_columna == max_nulos].index.tolist()

# Guardar los datos de esas columnas en un nuevo DataFrame
datos_columnas_mas_nulos = datos[columnas_mas_nulos]

# Imprimir resultados
print(f"Las columnas con más valores NaN ({max_nulos} valores nulos) son: {', '.join(columnas_mas_nulos)}\n")

# Convertir automáticamente "id" en categórico si es único en todas las filas
if datos["id"].nunique() == len(datos):  
    datos["id"] = datos["id"].astype(str)

if datos["availability_365"].nunique() < 365:
    datos["availability_365"] = datos["availability_365"].astype(str)


categoricas = datos.select_dtypes(include=['object', 'category', 'datetime']).columns.tolist()
numericas = datos.select_dtypes(include=['int64', 'float64']).columns.tolist()
total_categoricas = len(categoricas)
total_numericas = len(numericas)

print(f"En cuanto a tipos de datos tenemos:")
print(f"- {total_categoricas} Características Categóricas: {categoricas}")
print(f"- {total_numericas} Características Numéricas: {numericas}")

columnas_con_duplicados = [col for col in categoricas if datos[col].duplicated().sum() > 0]
print(f"Las siguientes columnas tienen valores duplicados: {columnas_con_duplicados}\n")

print(f"El número de registros duplicados en Name es: {datos['name'].duplicated().sum()}")
print(f"El número de registros duplicados en Host ID es: {datos['host_id'].duplicated().sum()}")
print(f"El número de registros duplicados en ID es: {datos['id'].duplicated().sum()}")

datos.drop(["id", "name", "host_name", "last_review", "reviews_per_month"], axis = 1, inplace = True)
datos.head()

fig, axis = plt.subplots(2, 3, figsize=(17, 10))

# Creacion del Histogram
sns.histplot(ax = axis[0,0], data = datos, x = "host_id")
sns.histplot(ax = axis[0,1], data = datos, x = "neighbourhood_group")
sns.histplot(ax = axis[0,2], data = datos, x = "neighbourhood").set_xticks([])
sns.histplot(ax = axis[1,0], data = datos, x = "room_type")
sns.histplot(ax = axis[1,1], data = datos, x = "availability_365")
fig.delaxes(axis[1, 2])

# Layout y enseñar la graficota
plt.tight_layout()
plt.show()

fig, axis = plt.subplots(4, 2, figsize=(10, 14), gridspec_kw={"height_ratios": [6, 1, 6, 1]})
variables = ["price", "minimum_nights", "number_of_reviews", "calculated_host_listings_count"]
# Graficar histogramas y boxplots en un bucle
for i, var in enumerate(variables):
    sns.histplot(ax=axis[i, 0], data=datos, x=var)
    sns.boxplot(ax=axis[i, 1], data=datos, x=var)
# Ajustar límites solo para 'minimum_nights'
axis[1, 0].set_xlim(0, 200)
plt.tight_layout()
plt.show()

fig, axis = plt.subplots(4, 2, figsize = (10, 16))
sns.regplot(ax = axis[0, 0], data = datos, x = "minimum_nights", y = "price")
sns.heatmap(datos[["price", "minimum_nights"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)
sns.regplot(ax = axis[0, 1], data = datos, x = "number_of_reviews", y = "price").set(ylabel = None)
sns.heatmap(datos[["price", "number_of_reviews"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])
sns.regplot(ax = axis[2, 0], data = datos, x = "calculated_host_listings_count", y = "price").set(ylabel = None)
sns.heatmap(datos[["price", "calculated_host_listings_count"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 0]).set(ylabel = None)
fig.delaxes(axis[2, 1])
fig.delaxes(axis[3, 1])
plt.tight_layout()
plt.show()

fig, axis = plt.subplots(figsize = (10, 8))
sns.countplot(data = datos, x = "room_type", hue = "neighbourhood_group")
plt.show()

datos["room_type"] = pd.factorize(datos["room_type"])[0]
datos["neighbourhood_group"] = pd.factorize(datos["neighbourhood_group"])[0]
datos["neighbourhood"] = pd.factorize(datos["neighbourhood"])[0]

fig, axes = plt.subplots(figsize=(15, 15))
sns.heatmap(datos[["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights",	
                        "number_of_reviews", "calculated_host_listings_count", "availability_365"]].corr(), annot = True, fmt = ".2f")
plt.tight_layout()
plt.show()
print(f"Aqui analizariamos todos los datos a la vez\n")
sns.pairplot(data = datos)
datos.describe()


fig, axes = plt.subplots(3, 3, figsize=(15, 15))
# Lista de variables a graficar
variables = ["neighbourhood_group", "price", "minimum_nights", 
             "number_of_reviews", "calculated_host_listings_count", "availability_365", 
             "room_type"]
# Generar los boxplots automáticamente
for ax, var in zip(axes.flat, variables):
    sns.boxplot(ax=ax, data=datos, y=var)
# Ajustar diseño para evitar solapamientos
plt.tight_layout()
plt.show()

price_stats = datos["price"].describe()
price_stats


# Calcular el rango intercuartílico (IQR) del precio
price_iqr = price_stats["75%"] - price_stats["25%"]
# Definir los límites superior e inferior para detectar valores atípicos
limite_superior = price_stats["75%"] + 1.5 * price_iqr
limite_inferior = price_stats["25%"] - 1.5 * price_iqr
# Imprimir los resultados
print(f"Los límites superior e inferior para detectar valores atípicos son {round(limite_superior, 2)} y {round(limite_inferior, 2)}, con un rango intercuartílico de {round(price_iqr, 2)}")
# Eliminar valores atípicos donde el precio sea menor o igual a 0
datos = datos[datos["price"] > 0]
count_0 = datos[datos["price"] == 0].shape[0]  # Cantidad de registros con precio 0
count_1 = datos[datos["price"] == 1].shape[0]  # Cantidad de registros con precio 1
print("Cantidad de registros con precio 0: ", count_0)
print("Cantidad de registros con precio 1: ", count_1)



nights_stats = datos["minimum_nights"].describe()
nights_stats
# Rango intercuartílico (IQR) para minimum_nights
nights_iqr = nights_stats["75%"] - nights_stats["25%"]
limite_superior = nights_stats["75%"] + 1.5 * nights_iqr
limite_inferior = nights_stats["25%"] - 1.5 * nights_iqr
print(f"Los límites superior e inferior para encontrar valores atípicos son {round(limite_superior, 2)} y {round(limite_inferior, 2)}, con un rango intercuartílico de {round(nights_iqr, 2)}")
datos = datos[datos["minimum_nights"] <= 15]
count_0 = datos[datos["minimum_nights"] == 0].shape[0]
count_1 = datos[datos["minimum_nights"] == 1].shape[0]
count_2 = datos[datos["minimum_nights"] == 2].shape[0]
count_3 = datos[datos["minimum_nights"] == 3].shape[0]
count_4 = datos[datos["minimum_nights"] == 4].shape[0]
print("Cantidad de 0: ", count_0)
print("Cantidad de 1: ", count_1)
print("Cantidad de 2: ", count_2)
print("Cantidad de 3: ", count_3)
print("Cantidad de 4: ", count_4)


review_stats = datos["number_of_reviews"].describe()
review_stats
# Rango intercuartílico (IQR) para number_of_reviews
review_iqr = review_stats["75%"] - review_stats["25%"]
limite_superior = review_stats["75%"] + 1.5 * review_iqr
limite_inferior = review_stats["25%"] - 1.5 * review_iqr
print(f"Los límites superior e inferior para encontrar valores atípicos son {round(limite_superior, 2)} y {round(limite_inferior, 2)}, con un rango intercuartílico de {round(review_iqr, 2)}")



hostlist_stats = datos["calculated_host_listings_count"].describe()
hostlist_stats
# Rango intercuartílico (IQR) para calculated_host_listings_count
hostlist_iqr = hostlist_stats["75%"] - hostlist_stats["25%"]
limite_superior = hostlist_stats["75%"] + 1.5 * hostlist_iqr
limite_inferior = hostlist_stats["25%"] - 1.5 * hostlist_iqr
print(f"Los límites superior e inferior para encontrar valores atípicos son {round(limite_superior, 2)} y {round(limite_inferior, 2)}, con un rango intercuartílico de {round(hostlist_iqr, 2)}")
count_04 = sum(1 for x in datos["calculated_host_listings_count"] if x in range(0, 5))
count_1 = datos[datos["calculated_host_listings_count"] == 1].shape[0]
count_2 = datos[datos["calculated_host_listings_count"] == 2].shape[0]
print("Cantidad de 0: ", count_04)
print("Cantidad de 1: ", count_1)
print("Cantidad de 2: ", count_2)
# Limpiar los valores atípicos
datos = datos[datos["calculated_host_listings_count"] > 4]


datos.isnull().sum().sort_values(ascending = False)
num_variables = ["number_of_reviews", "minimum_nights", "calculated_host_listings_count", 
                 "availability_365", "neighbourhood_group", "room_type"]
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(datos[num_variables])
df_scal = pd.DataFrame(scal_features, index = datos.index, columns = num_variables)
df_scal["price"] = datos["price"]
df_scal.head()



X = df_scal.drop("price", axis = 1)
y = df_scal["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
selection_model = SelectKBest(chi2, k = 4)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])
X_train_sel.head()


X_train_sel["price"] = list(y_train)
X_test_sel["price"] = list(y_test)
X_train_sel.to_csv("/workspaces/machine-learning-python-template/data/processed/clean_train.csv", index = False)
X_test_sel.to_csv("/workspaces/machine-learning-python-template/data/processed/clean_test.csv", index = False)