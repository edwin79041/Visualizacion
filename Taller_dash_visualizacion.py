#!/usr/bin/env python
# coding: utf-8

# # Taller Visualización para la analítica de datos
# 
# 
# Actividad:
# 
# En la base de datos de la Agencia Nacional de Contratación Pública Colombia Compra Eficiente, como se vio en uno de los primeros cuadernos del curso, está alojada la información de los contratos hechos por departamento en diferentes áreas (Salud, Educación, etc.). El objetivo de este taller es que **analice el comportamiento de la inversión en tales contratos por ciudad, departamento o región.** Su primera tarea con dicha base de datos, será hacer una exploración profunda, de tal manera que usted pueda identificar algunos gráficos claves en los cuales muestre hechos relevantes sobre el ítem a estudiar. A medida que usted conoce la base de datos será capaz de generar algunos indicadores de desempeño que intervengan en la inversión en los contratos en las diferentes locaciones, explique cada uno de sus indicadores y justifique su importancia. Genere una base de datos en la que se encuentren los elementos necesarios para calcular sus indicadores de desempeño. Genere clusters, mediante un modelo, para clasificar los departamentos.
# 
# Entregables:
# 1.	Dashboard en el que se evidencie su tarea de exploración.
# 2.	KPI’s con sus respectivas explicaciones y justificaciones.
# 3.	Visualización del modelo dentro del Dashboard (puede ser un mapa de Colombia pintado según su modelo, es una sugerencia más no una camisa de fuerza, sienta plena libertad de hacer lo que usted considere adecuado).
# 4.	Fecha de Entrega: 
# 

# In[1]:


#Importacion de librerias

import numpy                 as np
import pandas                as pd
import matplotlib.pyplot     as plt
import seaborn               as sns
import pandas                as pd
import matplotlib.pyplot     as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import interact
import ipywidgets as widgets
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from jupyter_dash import JupyterDash
import dash
from dash.dependencies import Input, Output
import sklearn.metrics       as Metrics
from scipy import stats

#warnings.simplefilter("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[2]:


#Importar la Base
df = pd.read_csv('SECOP_II_-_Contratos_Electr_nicos.csv')


# In[3]:


#Reseteamos el index y creamos una variable con un 1 en toda la base
df=df.reset_index(drop=True)


# In[4]:


df.tail()


# In[5]:


#Descriptivas de la base
df.info(null_counts=None,)


# In[6]:


df.columns


# In[7]:


#Transformacion de variables(Departamento)
df.loc[df['Departamento'] == 'Distrito Capital de Bogotá', 'Departamento'] = 'Bogotá'
df.loc[df['Departamento'] == 'San Andrés, Providencia y Santa Catalina', 'Departamento'] = 'San Andrés'
df['Departamento'].unique()


# In[8]:


df['Sector'].unique()


# In[9]:


#Transformacion de variables(Sector)
df.loc[df['Sector'] == 'Tecnologías de la Información y las Comunicaciones', 'Sector'] = 'TIC'
df.loc[df['Sector'] == 'Ambiente y Desarrollo Sostenible', 'Sector'] = 'Ambiente y Desarrollo'
df.loc[df['Sector'] == 'Inteligencia Estratégica y Contrainteligencia', 'Sector'] = 'Inteligencia Estratégica'


# In[10]:


#Se selecciona orden territorial para el análisis (excluyendo nacional), además los contratos con valores iguales a cero y no reversados
df2=df[(df['Orden'] == 'Territorial') & (df['Valor del Contrato'] > 0) & (df['Reversion']=='No') & (df['Sector']!='No aplica/No pertenece')]
df2


# In[11]:


df2['Departamento'].unique()


# In[12]:


df2['Sector'].unique()


# In[13]:


## Tabla de cantidad de población por departamento
poblac = pd.read_excel('departamentos.xlsx')
poblac


# In[14]:


df2.head(4)


# In[15]:


#Seleccionamos las variables necesarias para el análisis
df4=df2[['Departamento','Sector','Fecha de Inicio de Ejecucion','Fecha de Fin de Ejecucion','Saldo CDP','Valor del Contrato']]
df4


# In[16]:


df4['Fecha de Fin de Ejecucion']=pd.to_datetime(df4['Fecha de Fin de Ejecucion'],errors = 'coerce')
df4['Fecha de Inicio de Ejecucion']=pd.to_datetime(df4['Fecha de Inicio de Ejecucion'],errors = 'coerce')


# In[17]:


#
df4['Vigencia']=df4['Fecha de Fin de Ejecucion']-df4['Fecha de Inicio de Ejecucion']
df4


# In[18]:


df4 = df4.drop(['Fecha de Fin de Ejecucion', 'Fecha de Inicio de Ejecucion'], axis=1)


# In[19]:


df4.dtypes


# In[20]:


df4['Vigencia'] = (df4['Vigencia'] / np.timedelta64(1,'D'))


# In[21]:


df4 = df4.fillna(0)


# In[22]:


df4


# In[23]:


df4['conteo']=1


# In[24]:


# df4['Vigencia']+df4['Vigencia'].astype(int)
df4.dtypes


# In[25]:


df4 = df4.dropna()
df4


# Ajuste para gráfica con fechas
# 

# In[26]:


#Agrupar las variables cuantitativas
df5 = df4.groupby(['Departamento', 'Sector'])['Saldo CDP','Valor del Contrato','Vigencia','conteo'].sum()
df5.reset_index(inplace=True)
df5


# In[27]:


modelo = df5.merge(poblac,how='left',left_on='Departamento',right_on='Departamento')
modelo['dfpgal_Percapita'] =modelo['Valor del Contrato']/modelo['poblacion']
modelo=modelo.dropna()
modelo.reset_index(inplace=True)
modelo=modelo.drop(['index'],axis=1)
modelo


# In[28]:


#Agrupar las variables por departamento
agrupa_dptos = df5.groupby(['Departamento'])['Saldo CDP','Valor del Contrato','Vigencia','conteo'].sum()
agrupa_dptos.reset_index(inplace=True)
agrupa_dptos


# In[29]:


#Agrupar las variables por sectores
agrupa_sector = df5.groupby(['Sector'])['Saldo CDP','Valor del Contrato','Vigencia','conteo'].sum()
agrupa_sector.reset_index(inplace=True)
agrupa_sector


# In[30]:


agrupa_dptos1 = agrupa_dptos.merge(poblac,how='left',left_on='Departamento',right_on='Departamento')
agrupa_dptos['dfpgal_Percapita'] =agrupa_dptos['Valor del Contrato']/agrupa_dptos1['poblacion']
agrupa_dptos1=agrupa_dptos1.dropna()
agrupa_dptos1


# ### Modelo K-meas

# In[31]:


from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy import cluster
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
get_ipython().system('pip install prince')
import prince
import warnings
warnings.simplefilter("ignore")


# In[32]:


#seleccionamos las varibles cualitativas
cuali=modelo.select_dtypes(include=["object"])
cuali


# In[33]:


cuanti=modelo.select_dtypes(include=[np.number])
cuanti = cuanti.fillna(0)
cuanti


# In[34]:


cuanti = cuanti.drop(['Vigencia','Saldo CDP','dfpgal_Percapita'],axis=1)


# In[35]:


cuanti


# Realizar el proceso de estandarización de las variables:

# In[36]:


escala=StandardScaler(with_mean=True, with_std=True)
escala.fit(cuanti)
datosestan=escala.transform(cuanti)


# Aplicar la tecnica ACP para identificar las variables que tienen mayor impacto en el modelo, teniendo un porcentaje del 90% para traer las de mayor correlación:

# In[37]:


pca=PCA(0.9) 
pca.fit(datosestan)  ## Ajusto el PCA (valores, vectores, varianza)
nuevosACP=pca.transform(datosestan)
pca.explained_variance_ratio_  


# In[38]:


nuevosACP.shape


# In[39]:


acm=prince.MCA(n_components=12)
acm.fit(cuali)
nuevosACM=acm.fit_transform(cuali)
nuevosACM


# In[40]:


x=list(range(len(acm.explained_inertia_)))
y=np.cumsum(acm.explained_inertia_)
fig=px.scatter(x=x, y=y)
fig.show()


# In[41]:


## Uno los datos del ACP y del ACM
nuevos1=pd.DataFrame(nuevosACP, index=modelo.index) ## se crea el DataFrame, se mantiene el index que ya se quitaron los faltantes 
nuevos2=pd.DataFrame(nuevosACM, index=modelo.index) ## se crea el DataFrame, se mantiene el index que ya se quitaron los faltantes
acm_acp=pd.concat([nuevos1, nuevos2], axis=1)
acm_acp


# In[42]:


## Realizo el Dendrograma
plt.rcParams["figure.figsize"] = (20,10)
dendogram=sch.dendrogram(sch.linkage(acm_acp, method='ward',metric="euclidean"))


# In[43]:


enlace=sch.linkage(acm_acp, method='ward',metric="euclidean")
corte=cluster.hierarchy.cut_tree(enlace, n_clusters=3)
corte[:,0]


# In[44]:


## Adiciono la variable grupo a la base cuali y cuanti
cuanti["Grupo"]=corte[:,0]
cuali["Grupo"]=corte[:,0]


# In[45]:


## Chi cuadrados con las cuali
a=cuali.columns
pvalor=[]
for i in range(len(a)):
  tabla=pd.crosstab(cuali[a[i]], cuali["Grupo"])
  b,c,d,e=chi2_contingency(tabla)
  pvalor.append(c)
pvalor=pd.DataFrame(pvalor, index=a, columns=["Pvalue"])
pvalor.sort_values(["Pvalue"], ascending=True)


# In[46]:


## Grafico de codo
within= []
for k in range(1,20):
    kmeanModel = KMeans(n_clusters=k).fit(acm_acp)
    within.append(kmeanModel.inertia_)
fig=px.line(x=list(range(1,20)), y=within )
fig.show()


# In[47]:


## K-means
kmedias=KMeans(n_clusters=3).fit(acm_acp)
corte2 = pd.DataFrame(kmedias.labels_,columns = ['grupo2'])
corte2 
#ids = ({'client_id': [1, 2, 3, 4, 5, 6]}, columns = ['client_id'])


# In[48]:


kmedias.labels_


# In[49]:


#cuanti=cuanti.reset_index(name="Conteo")


# In[50]:


## Adiciono la variable grupo a la base cuali y cuanti
cuanti1 = pd.concat([cuanti,corte2],axis=1)
cuanti1


# In[51]:


cuanti1 = pd.concat([cuanti,corte2],axis=1)
tabla_f = pd.concat([cuali,cuanti1],axis=1)
tabla_f["($) millions"] = "$" + (tabla_f["Valor del Contrato"].astype(float)/1000000).astype(str) + "MM"
tabla_f


# In[52]:


total=KMeans(n_clusters=1).fit(acm_acp).inertia_ ## Varianza total
within=kmedias.inertia_  ### varianza dentro de los grupos
between=total-within  ## varianza entre grupos
print("Total=", total, "y Dentro =", within , "y Entre=", between)
print("CCI=", np.round(100*between/total, 2), "%")  ## Coeficiente de correlación intraclase


# In[53]:


features = acm_acp.columns

components = pca.fit_transform(acm_acp)

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

fig_table = px.scatter(components, x=0, y=1, color=kmedias.labels_.astype(str),title='Distribución Clústeres')

for i, feature in enumerate(features):
    fig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings[i, 0],
        y1=loadings[i, 1]
    )
    fig.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
        
    )
fig_table.show()


# In[54]:


tabla = tabla_f.drop(columns=['Grupo'])


# In[55]:


tabla


# In[56]:


tabla2 = tabla


# In[57]:


tabla=tabla.rename(columns={"grupo2": "grupo"})


# In[58]:


tabla=tabla.groupby(["Sector","grupo"]).size()
tabla=tabla.reset_index(name="Conteo")
tabla
Sector=px.bar(tabla, x=tabla.Sector, y=tabla.Conteo, facet_col=tabla.grupo, title='Detalle de grupos por sector')
Sector.show()


# In[59]:


tabla2


# In[60]:


tabla2=tabla2.rename(columns={"grupo2": "grupo"})


# In[61]:


tabla2=tabla2.groupby(["Departamento","grupo"]).size()
tabla2=tabla2.reset_index(name="Conteo")
tabla2
Depart_modelo=px.bar(tabla2, x=tabla2.Departamento, y=tabla2.Conteo, facet_col=tabla2.grupo, title='Detalle de grupos por departamento')
Depart_modelo.show()


# #### **Otras gráficas**

# In[62]:


agrupa_dptos1


# In[63]:


agrupa_sector


# In[64]:


agrupa_dptos1.columns


# In[65]:


agrupa_dptos1["($) millions"] = "$" + (agrupa_dptos1["Valor del Contrato"].astype(float)/1000000).astype(str) + "MM"
agrupa_dptos1["Poblation"] = ["{:,}".format(x) for x in agrupa_dptos1["poblacion"]]
agrupa_dptos1


# In[66]:


import math
# log_data= math.log10(agrupa_dptos)
agrupa_dptos["Valor del Contrato2"]= np.log10(agrupa_dptos["Valor del Contrato"])


# In[67]:


ax = agrupa_dptos.groupby(
    'Departamento'
).sum()[
    'dfpgal_Percapita'
].plot(
    kind='bar', 
    color='skyblue',
    figsize=(10,7),
    grid=True
)

ax.set_xlabel('Departamentos')
ax.set_ylabel('Cantidad de contratos')
ax.set_title('Cantidad de contratos por departamento')

plt.show()


# In[68]:


ax = agrupa_dptos.groupby(
    'Departamento'
).sum()[
    'Valor del Contrato2'
].plot(
    kind='bar', 
    color='skyblue',
    figsize=(10,7),
    grid=True
)

ax.set_xlabel('Departamentos')
ax.set_ylabel('Valor de contratos2')
ax.set_title('Valor de contratos por departamento')

plt.show()


# In[69]:


agrupa_dptos2 = df5.groupby(['Departamento','Sector'])['Saldo CDP','Valor del Contrato','Vigencia','conteo'].sum()
agrupa_dptos2["Valor del Contrato2"]= np.log10(agrupa_dptos2["Valor del Contrato"])
agrupa_dptos2=agrupa_dptos2.reset_index(drop=False)


# In[70]:


fig, axs = plt.subplots(figsize=(20,16))
axs=sns.boxplot(data=agrupa_dptos2,y='Valor del Contrato2',x="Sector", )
axs.set_xticklabels(axs.get_xticklabels(), rotation=90);


# In[71]:


ax = sns.catplot(y="Departamento", x="Valor del Contrato2", kind='box',  data=agrupa_dptos2, aspect=2, orient="h")


# In[72]:


sns.relplot(x="Valor del Contrato2", y="conteo", hue='Sector', data=agrupa_dptos2 , size=15).set(title='Relación valor y cantidad de contratos')


# In[73]:


my_range=range(1,len(agrupa_dptos.index)+1)
fig,ax=plt.subplots(figsize=(12,10))
# The vertcval plot is made using the hline function
# I load the seaborn library only to benefit the nice looking feature
ax.hlines(y=my_range, xmin=0, xmax=agrupa_dptos['dfpgal_Percapita'],  alpha=0.4)
ax.scatter(agrupa_dptos['dfpgal_Percapita'], my_range, alpha=1)
 
# Add title and exis names
plt.yticks(my_range, agrupa_dptos['Departamento'])
plt.title("Valor percapita por Departamento", loc='left')
plt.xlabel('Total')
plt.ylabel('País')
plt.grid()
plt.show()


# In[74]:


import dash
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# In[75]:


tabla_f.head()


# In[76]:


cualicolumns=df2.select_dtypes(exclude=['int64','float64']).columns
depar= list(agrupa_dptos1['Departamento'])


# In[77]:


sector_resum = go.Figure(
    data=[go.Box(x=agrupa_dptos2["Sector"], y=agrupa_dptos2["Valor del Contrato2"], marker_color='tomato')],
    layout=go.Layout(
        title=go.layout.Title(text="Valor contratos por Sector")
    )
)
sector_resum.show()


# In[78]:


deps = go.Figure(
    data=[go.Box(x=agrupa_dptos2["Departamento"], y=agrupa_dptos2["Valor del Contrato2"])],
    layout=go.Layout(
        title=go.layout.Title(text="Valor contratos por Departamento")
    )
)
deps.show()


# In[79]:


# fig2=sns.relplot(x="Valor del Contrato", y="conteo", hue='Sector', data=agrupa_sector , size=15).set(title='Relación valor y cantidad de contratos')
app = JupyterDash(__name__,external_stylesheets=['https://bootswatch.com/5/simplex/bootstrap.css'])
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#app = JupyterDash(__name__,external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.H1("Exploracion de datos contratos en Colombia"),
    html.Div([
        html.H6("Contratación en colombia por Departamentos"),
        html.P("A continuación se puede encontrar el listado de departamentos de Colombia, el vólumen de contratos y el valor acumulado:"),
        dcc.Dropdown(
                    id='my-input',
                    options=[{'label': i, 'value': i} for i in depar],
                    value='Amazonas'),
    ],style={"width": "100%",
                  "height": "50%",
                  "justify-content": "center"}),
    html.Div([
        dcc.Graph(id='Grafica_dep'),
        dcc.Graph(id='my-output'),],
        style={"width": "100%",
                  "justify-content": "center"}),
    html.P("Los departamentos con mayor inversión en contratos son Bogóta, Caldas y Cordoba, sin embargo existen valores de contratos atípicos como Risaralda, Huila, Antioquia y Santander."), 
    html.P("Los departamentos con menor inversión en contratos son Amazonas, Vichada, San Andres y Guaviare, comunidades que requieren mas inversión en educación, salud y seguridad para su desarrollo."),
    html.Div([
        #html.H6("Segundo titulo"),
         dcc.Graph(figure=deps),
#          dcc.Graph(figure=plt)
         dcc.Graph(figure=sector_resum)
     ],
    
        style={ "height": "calc(90% - 10px)"
              }),
    html.P("Los sectores con mayor inversión en contratos son Servicio Publico, Hacienda y crédito publico e Inclusión sociales y reconciliación."),
    html.P("Los sectores con menor inversión en contratos son Trabajo, Ciencia y Tecnología, Justicia."),
        ])
@app.callback(
    Output(component_id='my-output', component_property='figure'),
    [Input(component_id='my-input', component_property='value')]
)

def figura(i):
    fig=0
    Freq=tabla_f[tabla_f['Departamento']==i]
    fig = go.Figure(data=[go.Table(
                columnwidth = [100,50,50],
                header=dict(values=['Sector','Valor del Contrato','Contratos'],
                            fill_color='paleturquoise',
                            align='center'),
                cells=dict(values=[Freq['Sector'],Freq['($) millions'],Freq['conteo']],
                           fill_color='lavender',
                           align='center'))
                                 ])
    fig.update_layout(height = 230)
    

    return fig
@app.callback(
    Output(component_id='Grafica_dep', component_property='figure'),
    [Input(component_id='my-input', component_property='value')]
)
           
def actualiza(input_value):
    fig2 = 0
    Freq=tabla_f[tabla_f['Departamento']==input_value]
    fig2 = make_subplots(rows=1, cols=2)
    fig2.add_trace(go.Bar(x=Freq['Sector'], y=Freq['Valor del Contrato'], name = 'Valor de Contratos'), row=1, col=1)
    fig2.add_trace(go.Bar(x=Freq['Sector'], y=Freq['conteo'], name = 'Cantidad de Contratos'), row=1, col=2)
    fig2.update_layout(title_text="",
                      title_font_size=30,title_x=0.5)
    fig2.update_layout(height = 500)
    return fig2

app.run_server(debug=True,host="0.0.0.0",port=8050)


# In[80]:


app = JupyterDash(__name__,external_stylesheets=['https://bootswatch.com/5/simplex/bootstrap.css'])
#app = JupyterDash(__name__,external_stylesheets=external_stylesheets)

grupo = list(tabla_f['grupo2'].unique())


app.layout = html.Div(children=[
    html.Div(
        children=[
            html.Div(
                children=[html.H1("Resultado Cluster modelo K-means", style={"font-size": "30px"})], 
                
                style={
                  "display": "flex",
                  "justify-content": "center",
                  "height": "10%"
                })
        ],
    ),
    html.Div(
        children=[
            html.Div(
                children=[
                    html.P("Para la aplicación del modelo K-means se tuvieron en cuenta las siguientes variables: Departamento, Sector, Valor del Contrato, cantidad contratos, población."),                    
                    html.P("Por medio de la tecnica PCA se agrupan las caracteristicas mencionadas y se reducen los datos para el análisis, la cantidad resultante representa los datos originales."),                    
                    dcc.Graph(figure=fig_table),       ###gráfica 1
                    
                    html.Hr(),
                    
                    html.H2("Detalle de Grupos por departamento"),
                    html.P("El grupo 2 es el predominante que agrupa  la mayoria de contratos por departamento, mientras que el en el grupo 0 estan los departamento de Anquioquia , Bogota y Risaralda los cuales tienen una mayor cantidad de contratos"),
                    dcc.Graph(figure=Depart_modelo), 
                    html.P("El grupo 2 es el predominante que agrupa  la mayoria de contratos por Sector donde se destacan Servicio Publico, Educación y Salud."),                    
                    dcc.Graph(figure=Sector),   ###gráfica 2
                    
                    dcc.Dropdown(
                    id='my-input',
                    options=[{'label': i, 'value': i} for i in grupo],
                    value=0),
                    dcc.Graph(id='grupos'),
                    
                ],
                
                style={
                  "width": "100%",
                  "height": "100%",
                  "justify-content": "center",
                }
            ),
        ],
        
        style={
              "height": "calc(90% - 10px)",
              "display": "flex",
        }
    )

    
], style={
  
  "width": "100vh",
  "height": "100%",
  "padding": "10px"
})

@app.callback(
    Output(component_id='grupos', component_property='figure'),
    [Input(component_id='my-input', component_property='value')]
)
def actualiza(input_value):
    fig2 = 0
    Freq=tabla_f[tabla_f['grupo2']==input_value]
    fig2 = make_subplots(rows=1, cols=2)
    fig2.add_trace(go.Bar(x=Freq['Sector'], y=Freq['conteo'], name = 'Sector'), row=1, col=2)
    fig2.add_trace(go.Bar(x=Freq['Departamento'], y=Freq['conteo'], name = 'Departamento'), row=1, col=1)
    fig2.update_layout(title_text="Departamento y Sector",
                      title_font_size=30,title_x=0.5)
    fig2.update_layout(height = 500)
    return fig2


app.run_server(debug=True,host="0.0.0.0",port=8051)


# In[81]:


agrupa_dptos2


# In[82]:


Freq=agrupa_dptos2["Sector"].value_counts()
fig3 = go.Figure(data=[go.Pie(labels=Freq.index, values=Freq.values,hole=.3,title=i)])
fig3


# In[83]:


df2['conteo']=1


# In[84]:


df10=df2[['Departamento','Saldo CDP','Valor del Contrato','Valor Pendiente de Ejecucion','conteo']]


# In[85]:


df20=df2[['Departamento','Sector','Saldo CDP','Valor del Contrato','Valor Pendiente de Ejecucion','conteo']]


# In[86]:


#Agrupar las variables cuantitativas
df11 = df10.groupby(['Departamento'])['Saldo CDP','Valor del Contrato','Valor Pendiente de Ejecucion','conteo'].sum()
df11 = df11.merge(poblac,how='left',left_on='Departamento',right_on='Departamento')

df11 = df11.dropna()


# In[87]:


df11['Porcentaje_Ejecución'] = (df11['Valor Pendiente de Ejecucion'] / (df11['Valor del Contrato']))
df11['PerCapitalcontrato'] = (df11['Valor del Contrato'] / (df11['poblacion']))


# In[88]:


df11['Valor_Prom_Contrato'] = (df11['Valor del Contrato'] / (df11['conteo']))


# In[89]:


df20['Valor_Prom_Contrato'] = (df20['Valor del Contrato'] / (df20['conteo']))


# In[90]:


df20


# In[91]:


#df10.loc[1, df10Porcentaje_Ejecución] = 0
df11['Porcentaje_Ejecución1'] = df11['Porcentaje_Ejecución'].replace([1], 0)


# In[92]:


df11


# In[93]:


df20


# In[94]:


app = JupyterDash(__name__,external_stylesheets=['https://bootswatch.com/5/simplex/bootstrap.css'])
#app = JupyterDash(__name__,external_stylesheets=external_stylesheets)
depart = list(df20['Departamento'].unique())
app.layout = html.Div(children=[
    html.Div(
        children=[
            html.Div(
                children=[html.H1("Kpi´s", style={"font-size": "30px"})],          
                style={
                  "display": "flex",
                  "justify-content": "center",
                  "height": "10%"
                })
        ],
    ),
    html.Div(
        children=[
            html.Div(
                children=[
                    dcc.Dropdown(
                    id='my-input',
                    options=[{'label': i, 'value': i} for i in depart],
                    value='Amazonas'),
                    html.P("En la siguiente gráfica se encuentra el detalle de la distribución por cantidad de contratos:"),
                    dcc.Graph(id='kpis'),   
                    html.Hr(),
                    html.P("A continuación se presenta la participación por departamento, el valor promedio del total de contratos y el indicador Percapita:"),
                    html.P("1. Porcentaje de ejecución: Valor pendiente de ejecución / valor total del contrato"),
                    html.P("2. Valor promedio total de contratos: Valor total de contratos / cantidad de contratos adjudicados"),
                    html.P("3. Valor percapita de contratos: Valor total de contratos / Población del Departamento"),
                    dcc.Graph(id='tabla'),       ###gráfica 1
       
                ],
                style={
                  "width": "100%",
                  "height": "100%",
                  "justify-content": "center",
                }
            ),
        ],
        style={
              "height": "calc(90% - 10px)",
              "display": "flex",
        }
    )

    
], style={
  
  "width": "100vh",
  "height": "100%",
  "padding": "10px"
})
@app.callback(
    Output(component_id='kpis', component_property='figure'),
    [Input(component_id='my-input', component_property='value')]
)
def kpis(input_value):
    
    depa = df20[df20['Departamento']==input_value]
    fig3 = px.sunburst(depa, path=['Sector'], values='conteo')

    return fig3

@app.callback(
    Output(component_id='tabla', component_property='figure'),
    [Input(component_id='my-input', component_property='value')]
)
def tablakpi(input_value):
    fig5=0
    tablakpi1=df11[df11['Departamento']==input_value]
    fig5 = go.Figure(data=[go.Table(
                columnwidth = [100,150,150],
                header=dict(values=['Porcentaje de Ejecución(%)','Valor promedio por contrato','Valor percapita de los contratos'],
                            fill_color='paleturquoise',
                            align='center'),
                cells=dict(values=[tablakpi1['Porcentaje_Ejecución']*100,tablakpi1['Valor_Prom_Contrato'],tablakpi1['PerCapitalcontrato'],],
                           fill_color='lavender',
                           align='center'))
                                 ])
    fig5.update_layout(height = 280)

    return fig5


app.run_server(debug=True,host="0.0.0.0",port=8888)

