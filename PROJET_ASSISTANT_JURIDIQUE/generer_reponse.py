import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)

CLE_API = "AIzaSyBMVSxIv6JEERdwpZ7ZBSYENy2Wgv1EJEc"
genai.configure(api_key=CLE_API)

modele = genai.GenerativeModel(
    "models/gemini-1.5-flash",
    generation_config={
        "temperature": 0.3,
        "max_output_tokens": 2048,
    }
)

def preparer_contexte(documents, limite_tokens=6000):
    contexte_parts = []
    tokens_actuels = 0
    
    for doc in documents:
        contenu = doc.page_content.strip()
        tokens_estimation = len(contenu.split())
        
        if tokens_actuels + tokens_estimation > limite_tokens:
            break
            
        contexte_parts.append(contenu)
        tokens_actuels += tokens_estimation
    
    return "\n\n---\n\n".join(contexte_parts)

def generer_reponse(question, documents):
    if not documents:
        return "Aucun document pertinent trouvé pour répondre à votre question."
    
    try:
        contexte = preparer_contexte(documents)
        
        if not contexte.strip():
            return "Les documents trouvés ne contiennent pas d'information exploitable."
        
        prompt = f"""Tu es un assistant juridique spécialisé dans le droit du travail ivoirien.

EXTRAITS DE LOIS PERTINENTS :
{contexte}

QUESTION : {question}

INSTRUCTIONS :
1. Analyse les extraits fournis
2. Si les extraits contiennent des informations pertinentes, fournis une réponse claire et précise, avec un language naturel et simple pour les non expert
3. Si les informations sont insuffisantes, indique : "Les documents fournis ne contiennent pas d'information suffisante pour répondre complètement à cette question."
4. Cite les articles ou références légales quand c'est possible
5. Réponds en français simple et accessible

RÉPONSE :"""

        reponse = modele.generate_content(prompt)
        return reponse.text.strip()
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération de réponse: {e}")
        return "Erreur technique lors de la génération de la réponse. Veuillez réessayer."

def generer_reponse_avec_source(question, documents):
    reponse_principale = generer_reponse(question, documents)
    
    if documents and "ne contiennent pas d'information suffisante" not in reponse_principale:
        sources = set()
        for doc in documents[:5]:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                nom_fichier = doc.metadata['source'].split('/')[-1]
                sources.add(nom_fichier)
        
        if sources:
            sources_text = ", ".join(sorted(sources))
            reponse_principale += f"\n\n📄 Sources consultées : {sources_text}"
    
    return reponse_principale
