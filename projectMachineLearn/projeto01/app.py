import streamlit as st  
import pandas as pd 
from sklearn.linear_model import LinearRegression

df = pd.read_csv('pizzas.csv')


modelo = LinearRegression()
x = df[["diametro"]]
y = df[["preco"]] 
modelo.fit(x, y) 

st.title("Modelo de Regressão Linear para Previsão de Preço de Pizza")
st.divider()

diametro = st.number_input("Digite o diâmetro da pizza")


if diametro:
    preco_previsto = modelo.predict([[diametro]])
    st.write(f"O preço previsto para uma pizza de {diametro:.2f} cm é: R$ {preco_previsto[0][0]:.2f}")
    st.balloons()