from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from evaluateur import EvaluateurCRAG, ReformulateursRequete
import logging
import os
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class MoteurRechercheCRAG:
    def __init__(self, chemin_index="base_vecteurs", cle_api=None):
        try:
          
            if cle_api is None:
                cle_api = os.getenv('GOOGLE_API_KEY', "AIzaSyBMVSxIv6JEERdwpZ7ZBSYENy2Wgv1EJEc")
            
        
            model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'batch_size': 1}
            )
            
          
            if not os.path.exists(chemin_index):
                raise FileNotFoundError(f"Index non trouvé: {chemin_index}. Exécutez d'abord creer_vecteurs.py")
            
            self.base = FAISS.load_local(chemin_index, model, allow_dangerous_deserialization=True)
            self.evaluateur = EvaluateurCRAG(cle_api)
            self.reformulateur = ReformulateursRequete(cle_api)
            self.historique_recherches = {}
            
            logger.info(" Moteur CRAG initialisé avec succès")
        except Exception as e:
            logger.error(f" Erreur initialisation CRAG: {e}")
            raise

    def recherche_crag(self, question: str, k: int = 15, max_iterations: int = 3) -> Dict:
        """Méthode principale CRAG avec logique itérative"""
        resultats_finaux = {
            'documents': [],
            'score_confiance': 0.0,
            'strategie_utilisee': '',
            'iterations': 0,
            'evaluations': []
        }
        
        question_courante = question
        iteration = 0
        
        logger.info(f" Début recherche CRAG: {question[:50]}...")
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f" CRAG Itération {iteration}: {question_courante[:50]}...")
            
        
            documents_bruts = self._recherche_vectorielle(question_courante, k)
            
            if not documents_bruts:
                logger.warning(f"Aucun document trouvé à l'itération {iteration}")
                if iteration == 1:
           
                    documents_bruts = self._recherche_etendue(question, k * 2)
                
                if not documents_bruts:
                    break
            
          
            evaluation = self.evaluateur.evaluer_pertinence(question, documents_bruts)
            resultats_finaux['evaluations'].append(evaluation)
            
            logger.info(f" Décision CRAG: {evaluation['decision']} (Score: {evaluation['score_global']:.2f})")
            
            
            if evaluation['decision'] == 'UTILISER':
                resultats_finaux.update({
                    'documents': evaluation['documents_pertinents'],
                    'score_confiance': evaluation['score_global'],
                    'strategie_utilisee': f'DIRECT (iteration {iteration})',
                    'iterations': iteration
                })
                break
                
            elif evaluation['decision'] == 'UTILISER_PARTIEL':
                if iteration == max_iterations or evaluation['score_global'] > 0.6:
                    resultats_finaux.update({
                        'documents': evaluation['documents_pertinents'],
                        'score_confiance': evaluation['score_global'],
                        'strategie_utilisee': f'PARTIEL (iteration {iteration})',
                        'iterations': iteration
                    })
                    break
                else:
                    question_courante = self._reformuler_question(question, documents_bruts)
                    
            elif evaluation['decision'] == 'REFORMULER':
                question_courante = self._reformuler_question(question, documents_bruts)
                
            elif evaluation['decision'] == 'CHERCHER_PLUS':
              
                documents_etendus = self._recherche_etendue(question, k * 2)
                if documents_etendus:
                    evaluation_etendue = self.evaluateur.evaluer_pertinence(question, documents_etendus)
                    if evaluation_etendue['score_global'] > evaluation['score_global']:
                        resultats_finaux.update({
                            'documents': evaluation_etendue['documents_pertinents'],
                            'score_confiance': evaluation_etendue['score_global'],
                            'strategie_utilisee': f'RECHERCHE_ETENDUE (iteration {iteration})',
                            'iterations': iteration
                        })
                        break
                
               
                question_courante = self._reformuler_question(question, documents_bruts)
        
        
        if not resultats_finaux['documents'] and resultats_finaux['evaluations']:
            meilleure_eval = max(resultats_finaux['evaluations'], key=lambda x: x['score_global'])
            resultats_finaux.update({
                'documents': meilleure_eval['documents_pertinents'][:5],
                'score_confiance': meilleure_eval['score_global'],
                'strategie_utilisee': 'FALLBACK_MEILLEURE',
                'iterations': max_iterations
            })
        
        logger.info(f" CRAG terminé: {len(resultats_finaux['documents'])} documents, score {resultats_finaux['score_confiance']:.2f}")
        return resultats_finaux
    
    def _recherche_vectorielle(self, question: str, k: int) -> List:
        """Recherche vectorielle de base"""
        try:
            resultats = self.base.similarity_search_with_score(question, k=k)
            documents = []
            
            for doc, score in resultats:
         
                if score < 1.5:  
                    documents.append(doc)
            
            logger.info(f" Recherche vectorielle: {len(documents)} documents trouvés")
            return documents
        except Exception as e:
            logger.error(f" Erreur recherche vectorielle: {e}")
            return []
    
    def _reformuler_question(self, question_originale: str, documents_contexte: List) -> str:
        """Reformule la question pour améliorer la recherche"""
        try:
            variantes = self.reformulateur.generer_variantes(question_originale, 2)
            if variantes and len(variantes) > 0:
                nouvelle_question = variantes[0]
                logger.info(f" Question reformulée: {nouvelle_question[:50]}...")
                return nouvelle_question
        except Exception as e:
            logger.error(f" Erreur reformulation: {e}")
        
        return question_originale
    
    def _recherche_etendue(self, question: str, k: int) -> List:
        """Recherche étendue basée sur les mots-clés"""
        try:
            mots_cles = self.reformulateur.extraire_mots_cles_juridiques(question)
            tous_documents = []
            
         
            for mot_cle in mots_cles[:3]:  
                try:
                    docs_mot_cle = self.base.similarity_search(mot_cle, k=k//3)
                    tous_documents.extend(docs_mot_cle)
                except:
                    continue
            
            
            documents_uniques = []
            contenus_vus = set()
            
            for doc in tous_documents:
                contenu_signature = doc.page_content[:100] 
                if contenu_signature not in contenus_vus:
                    documents_uniques.append(doc)
                    contenus_vus.add(contenu_signature)
            
            logger.info(f" Recherche étendue: {len(documents_uniques)} documents uniques")
            return documents_uniques[:k]
            
        except Exception as e:
            logger.error(f" Erreur recherche étendue: {e}")
            return []
    
    def rechercher(self, question: str, k: int = 15) -> List:
        """Méthode compatible avec l'interface existante"""
        resultats_crag = self.recherche_crag(question, k)
        return resultats_crag['documents']
    
    def rechercher_avec_details(self, question: str, k: int = 15) -> Dict:
        """Méthode pour obtenir les détails complets CRAG"""
        return self.recherche_crag(question, k)


class MoteurRecherche(MoteurRechercheCRAG):
    """Classe wrapper pour compatibilité avec interface.py"""
    
    def __init__(self, chemin_index="base_vecteurs"):
        try:
            super().__init__(chemin_index)
            logger.info(" MoteurRecherche (interface compatible) initialisé")
        except Exception as e:
            logger.error(f" Erreur initialisation MoteurRecherche: {e}")
            
            self._initialiser_fallback(chemin_index)
    
    def _initialiser_fallback(self, chemin_index):
        """Mode fallback si CRAG ne fonctionne pas"""
        try:
            model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            self.base = FAISS.load_local(chemin_index, model, allow_dangerous_deserialization=True)
            self.evaluateur = None
            self.reformulateur = None
            logger.warning(" Mode fallback activé (recherche vectorielle simple)")
        except Exception as e:
            logger.error(f" Même le mode fallback a échoué: {e}")
            raise
    
    def rechercher(self, question: str, k: int = 15) -> List:
        """Recherche compatible avec l'interface"""
        try:
            if hasattr(self, 'evaluateur') and self.evaluateur is not None:
                
                return super().rechercher(question, k)
            else:
               
                logger.info(" Utilisation du mode fallback")
                resultats = self.base.similarity_search(question, k=k)
                return resultats[:10] 
        except Exception as e:
            logger.error(f" Erreur recherche: {e}")
            return []