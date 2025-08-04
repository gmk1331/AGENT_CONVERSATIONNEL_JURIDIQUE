import google.generativeai as genai


CLE_API = "AIzaSyBMVSxIv6JEERdwpZ7ZBSYENy2Wgv1EJEc"
genai.configure(api_key=CLE_API)

modele = genai.GenerativeModel("models/gemini-1.5-flash")

def generer_reponse(contexte, question):
    prompt = f"""
Tu es un assistant juridique spécialisé dans le droit du travail ivoirien.
Voici les extraits de loi extraits automatiquement :

{contexte}

En t'appuyant uniquement sur ces extraits :
- Donne une réponse claire, synthétique et juridiquement exacte à la question :
"{question}"

- Si les extraits ne permettent pas de répondre, indique : "Les documents fournis ne contiennent pas d'information suffisante."

Réponds en français accessible a tous , de maniere simple et precise.
"""
    reponse = modele.generate_content(prompt)
    return reponse.text.strip()
