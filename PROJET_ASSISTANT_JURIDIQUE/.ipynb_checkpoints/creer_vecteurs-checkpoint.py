import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def charger_pdfs(dossier="donnees"):
    documents = []
    for fichier in os.listdir(dossier):
        if fichier.endswith(".pdf"):
            chemin = os.path.join(dossier, fichier)
            loader = PyMuPDFLoader(chemin)
            textes = loader.load()
            documents.extend(textes)
    return documents

def decouper_textes(documents):
    separateur = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return separateur.split_documents(documents)

def construire_base_vecteurs(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vecteurs = FAISS.from_documents(chunks, embeddings)
    vecteurs.save_local("base_vecteurs")
    print(" Base sauvegardée dans 'base_vecteurs'")

if __name__ == "__main__":
    print(" Chargement des fichiers PDF...")
    docs = charger_pdfs()
    print(f" {len(docs)} documents chargés")

    print(" Découpage des textes...")
    morceaux = decouper_textes(docs)
    print(f" {len(morceaux)} morceaux générés")

    print(" Création de la base de vecteurs...")
    construire_base_vecteurs(morceaux)
