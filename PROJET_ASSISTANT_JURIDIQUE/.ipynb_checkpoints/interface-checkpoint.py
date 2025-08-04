import streamlit as st
from moteur_recherche import MoteurRecherche
from generer_reponse import generer_reponse



retriever = MoteurRecherche()
st.set_page_config(page_title="Assistant Juridique ACJ", layout="centered")
st.markdown("<h1 style='text-align: center;'> CHAT WITH ACJ</h1>", unsafe_allow_html=True)

if "historique" not in st.session_state:
    st.session_state.historique = []

st.subheader(" Historique de la conversation")
for chat in st.session_state.historique:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["reponse"])

question = st.chat_input("Posez votre question ici...")

if question:
    documents = retriever.rechercher(question)
    reponse = generer_reponse(question, documents)
    st.session_state.historique.append({
        "question": question,
        "reponse": reponse
    })
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(reponse)
