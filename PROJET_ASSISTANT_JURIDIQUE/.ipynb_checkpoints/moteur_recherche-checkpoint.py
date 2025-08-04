from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class MoteurRecherche:
    def __init__(self, chemin_index="base_vecteurs"):
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.base = FAISS.load_local(chemin_index, model, allow_dangerous_deserialization=True)

    def rechercher(self, question, k=10):
        return self.base.similarity_search(question, k=k)
