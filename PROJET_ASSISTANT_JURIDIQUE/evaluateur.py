import google.generativeai as genai
import logging
from typing import List, Dict, Tuple
import re

logger = logging.getLogger(__name__)

class EvaluateurCRAG:
    def __init__(self, cle_api: str):
        genai.configure(api_key=cle_api)
        self.modele = genai.GenerativeModel(
            "models/gemini-1.5-flash",
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 500,
            }
        )
    
    def evaluer_pertinence(self, question: str, documents: List) -> Dict:
        if not documents:
            return {
                'score_global': 0.0,
                'decision': 'CHERCHER_PLUS',
                'documents_pertinents': [],
                'raison': 'Aucun document trouvé'
            }
        
        contexte_evaluation = self._preparer_contexte_evaluation(documents[:5])
        
        prompt_evaluation = f"""Tu es un évaluateur de pertinence pour un système juridique.

QUESTION: {question}

DOCUMENTS À ÉVALUER:
{contexte_evaluation}

Évalue si ces documents contiennent des informations suffisamment pertinentes pour répondre à la question.

Réponds EXACTEMENT dans ce format:
SCORE: [0.0 à 1.0]
DECISION: [CORRECT|AMBIGU|INCORRECT]
RAISON: [explication courte]

Critères:
- CORRECT (0.8-1.0): Documents très pertinents, réponse complète possible
- AMBIGU (0.4-0.7): Informations partielles, réponse possible mais incomplète  
- INCORRECT (0.0-0.3): Documents non pertinents ou hors sujet
"""
        
        try:
            reponse = self.modele.generate_content(prompt_evaluation)
            return self._parser_evaluation(reponse.text, documents)
        except Exception as e:
            logger.error(f"Erreur évaluation: {e}")
            return {
                'score_global': 0.5,
                'decision': 'AMBIGU',
                'documents_pertinents': documents[:3],
                'raison': 'Erreur évaluation'
            }
    
    def _preparer_contexte_evaluation(self, documents: List) -> str:
        contexte_parts = []
        for i, doc in enumerate(documents, 1):
            contenu = doc.page_content[:300]
            contexte_parts.append(f"DOC{i}: {contenu}...")
        return "\n\n".join(contexte_parts)
    
    def _parser_evaluation(self, texte_reponse: str, documents: List) -> Dict:
        try:
            score_match = re.search(r'SCORE:\s*([\d.]+)', texte_reponse)
            decision_match = re.search(r'DECISION:\s*(\w+)', texte_reponse)
            raison_match = re.search(r'RAISON:\s*(.+?)(?:\n|$)', texte_reponse)
            
            score = float(score_match.group(1)) if score_match else 0.5
            decision_brute = decision_match.group(1) if decision_match else 'AMBIGU'
            raison = raison_match.group(1).strip() if raison_match else 'Évaluation automatique'
            
            if decision_brute == 'CORRECT':
                decision = 'UTILISER'
                docs_pertinents = documents
            elif decision_brute == 'AMBIGU':
                decision = 'REFORMULER' if score < 0.6 else 'UTILISER_PARTIEL'
                docs_pertinents = documents[:max(2, len(documents)//2)]
            else:
                decision = 'CHERCHER_PLUS'
                docs_pertinents = []
            
            return {
                'score_global': score,
                'decision': decision,
                'documents_pertinents': docs_pertinents,
                'raison': raison
            }
        except Exception as e:
            logger.error(f"Erreur parsing: {e}")
            return {
                'score_global': 0.5,
                'decision': 'UTILISER_PARTIEL',
                'documents_pertinents': documents[:3],
                'raison': 'Évaluation par défaut'
            }

class ReformulateursRequete:
    def __init__(self, cle_api: str):
        genai.configure(api_key=cle_api)
        self.modele = genai.GenerativeModel("models/gemini-1.5-flash")
    
    def generer_variantes(self, question_originale: str, nb_variantes: int = 3) -> List[str]:
        prompt = f"""Génère {nb_variantes} reformulations différentes de cette question juridique pour améliorer la recherche documentaire:

QUESTION ORIGINALE: {question_originale}

Génère des variantes qui:
1. Utilisent des synonymes juridiques
2. Changent la structure de la phrase
3. Ajoutent des termes techniques pertinents

Format: une reformulation par ligne, sans numérotation."""

        try:
            reponse = self.modele.generate_content(prompt)
            variantes = [ligne.strip() for ligne in reponse.text.split('\n') if ligne.strip()]
            return variantes[:nb_variantes]
        except Exception as e:
            logger.error(f"Erreur reformulation: {e}")
            return [question_originale]
    
    def extraire_mots_cles_juridiques(self, question: str) -> List[str]:
        prompt = f"""Extrait les mots-clés juridiques les plus importants de cette question:

QUESTION: {question}

Retourne uniquement les mots-clés séparés par des virgules, sans explication."""

        try:
            reponse = self.modele.generate_content(prompt)
            mots_cles = [mot.strip() for mot in reponse.text.split(',')]
            return [mot for mot in mots_cles if mot and len(mot) > 2]
        except Exception as e:
            logger.error(f"Erreur extraction mots-clés: {e}")
            return question.split()[:5]