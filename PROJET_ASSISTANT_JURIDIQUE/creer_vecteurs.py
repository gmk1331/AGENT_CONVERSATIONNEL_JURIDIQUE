import os
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def charger_pdf_individuel(chemin_fichier):
    """Charge un PDF individuel avec gestion d'erreurs robuste"""
    try:
        loader = PyMuPDFLoader(chemin_fichier)
        docs = loader.load()
        
        
        for doc in docs:
            doc.metadata['source'] = os.path.basename(chemin_fichier)
            doc.metadata['type'] = 'document_juridique'
            # Nettoyage du contenu
            doc.page_content = doc.page_content.strip()
        
        logger.info(f" Chargé: {os.path.basename(chemin_fichier)} - {len(docs)} pages")
        return docs
    except Exception as e:
        logger.error(f" Erreur lors du chargement de {chemin_fichier}: {e}")
        return []

def charger_pdfs(dossier="donnees", max_workers=3):
    """Charge tous les PDFs du dossier de manière parallèle"""
    if not os.path.exists(dossier):
        logger.error(f" Dossier '{dossier}' non trouvé!")
        return []
    
    fichiers_pdf = [os.path.join(dossier, f) for f in os.listdir(dossier) if f.endswith(".pdf")]
    
    if not fichiers_pdf:
        logger.error(f" Aucun PDF trouvé dans '{dossier}'!")
        return []
    
    logger.info(f" Trouvé {len(fichiers_pdf)} fichiers PDF à traiter")
    documents = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(charger_pdf_individuel, fichier): fichier for fichier in fichiers_pdf}
        
        for future in as_completed(futures):
            docs = future.result()
            documents.extend(docs)
            gc.collect()
    
    logger.info(f" Total: {len(documents)} pages chargées")
    return documents

def decouper_textes(documents, chunk_size=1000, chunk_overlap=200):
    """Découpe les textes en chunks optimisés pour les documents juridiques"""
   
    separateurs_juridiques = [
        "\n\nArticle ",  
        "\n\nSection ",  
        "\n\nChapitre ", 
        "\n\n",          
        "\n",          
        ". ",            
        " ",            
        ""
    ]
    
    separateur = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separateurs_juridiques,
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = separateur.split_documents(documents)
    
   
    chunks_filtres = []
    for chunk in chunks:
        if len(chunk.page_content.strip()) > 50:  
            chunks_filtres.append(chunk)
    
    logger.info(f" {len(chunks_filtres)} chunks générés (filtrage appliqué)")
    return chunks_filtres

def construire_base_vecteurs_par_lots(chunks, taille_lot=30):
    """Construit la base de vecteurs par lots pour gérer la mémoire"""
    try:
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'batch_size': 8,  
                'show_progress_bar': True
            }
        )
        
        base_vecteurs = None
        total_lots = (len(chunks) + taille_lot - 1) // taille_lot
        
        logger.info(f" Début de la création de {total_lots} lots")
        
        for i in range(0, len(chunks), taille_lot):
            lot_actuel = i // taille_lot + 1
            logger.info(f" Traitement du lot {lot_actuel}/{total_lots}")
            
            lot_chunks = chunks[i:i+taille_lot]
            
            try:
                if base_vecteurs is None:
                    base_vecteurs = FAISS.from_documents(lot_chunks, embeddings)
                    logger.info(f" Base initialisée avec {len(lot_chunks)} documents")
                else:
                    lot_vecteurs = FAISS.from_documents(lot_chunks, embeddings)
                    base_vecteurs.merge_from(lot_vecteurs)
                    logger.info(f" Fusionné {len(lot_chunks)} documents supplémentaires")
                    del lot_vecteurs
                
                gc.collect()
                
            except Exception as e:
                logger.error(f" Erreur lot {lot_actuel}: {e}")
                continue
        
        if base_vecteurs is None:
            raise Exception("Impossible de créer la base de vecteurs")
        

        os.makedirs("base_vecteurs", exist_ok=True)
        base_vecteurs.save_local("base_vecteurs")
        logger.info(" Base sauvegardée dans 'base_vecteurs'")
        
        return base_vecteurs
        
    except Exception as e:
        logger.error(f" Erreur lors de la création de la base: {e}")
        raise

def main():
    """Fonction principale d'indexation"""
    try:
        logger.info("CRÉATION DE L'INDEX JURIDIQUE CRAG ===")
        
        
        if not os.path.exists("donnees"):
            logger.error(" Dossier 'donnees' manquant!")
            return False
        
        pdfs = [f for f in os.listdir("donnees") if f.endswith(".pdf")]
        if not pdfs:
            logger.error(" Aucun PDF dans le dossier 'donnees'!")
            return False
        
        logger.info(f" {len(pdfs)} PDFs détectés pour indexation")
        
     
        logger.info(" Chargement des fichiers PDF...")
        docs = charger_pdfs()
        
        if not docs:
            logger.error(" Aucun document chargé!")
            return False
        
        logger.info(f" {len(docs)} documents chargés")
        
       
        logger.info(" Découpage des textes...")
        morceaux = decouper_textes(docs)
        
        if not morceaux:
            logger.error(" Aucun chunk généré!")
            return False
        
        logger.info(f" {len(morceaux)} morceaux générés")
        
     
        logger.info(" Création de la base de vecteurs...")
        base_vecteurs = construire_base_vecteurs_par_lots(morceaux)
        
        
        logger.info(" Test de l'index créé...")
        test_results = base_vecteurs.similarity_search("contrat de travail", k=3)
        logger.info(f" Test réussi: {len(test_results)} résultats trouvés")
        
       
        del docs, morceaux
        gc.collect()
        
        logger.info("INDEXATION TERMINÉE AVEC SUCCÈS ===")
        logger.info(" Vous pouvez maintenant utiliser votre agent CRAG!")
        return True
        
    except Exception as e:
        logger.error(f" Erreur critique: {e}")
        return False


if __name__ == "__main__":  
    success = main()
    if success:
        print("\n Indexation réussie! Votre agent CRAG est prêt à fonctionner.")
    else:
        print("\n Échec de l'indexation. Vérifiez les logs ci-dessus.")