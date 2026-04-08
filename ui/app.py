import streamlit as st

st.title("SEF - Signal Extraction Framework")

# Input
name = st.text_input("Bella!!!!!! come ti chiami????")

# Button
if st.button("Saluta"):
    st.write(f"Ciao {name}, tutto a posto bro?")

# Slider
age = st.slider("Quanti anni hai?", 0, 100, 21)
st.write(f"Hai {age} anni")
st.write("ciao")