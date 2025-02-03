# MRS
gruppo 9 FIA

 

# **AI per la Classificazione dei Tumori al Seno**  

Il tumore al seno può essere suddiviso in due categorie principali: **benigno** e **maligno**. Le cellule benigne tendono a rimanere localizzate, con una crescita controllata e non invasiva, mentre quelle maligne presentano un comportamento aggressivo, caratterizzato da una rapida proliferazione e dalla capacità di invadere i tessuti circostanti e formare metastasi. La corretta classificazione delle cellule tumorali è essenziale per una diagnosi precoce e per la scelta del trattamento più adeguato.  

La distinzione tra cellule benigne e maligne avviene attraverso l’analisi di diverse caratteristiche morfologiche e biologiche, come la forma e la dimensione delle cellule, la struttura del nucleo, il grado di adesione e il tasso di mitosi. Questi parametri vengono estratti da test citologici e rappresentano informazioni fondamentali per determinare la natura del tumore.  

### **Obiettivo del Progetto**  
Questo progetto si propone di sviluppare un modello di **machine learning** capace di classificare le cellule tumorali in **benigne** o **maligne** con elevata accuratezza. A tal fine, verrà utilizzato il dataset **Breast Cancer Wisconsin (Original)**, ampiamente impiegato nella ricerca per la classificazione dei tumori al seno. L’algoritmo implementato potrà essere utilizzato come supporto ai medici per migliorare la precisione e la rapidità della diagnosi.  

### **Metodologia**  
Per garantire una valutazione affidabile del modello, verranno applicate diverse tecniche di validazione, tra cui:  
- **K-Fold Cross Validation**  
- **Leave-One-Out Cross Validation (LOO-CV)**  

Il modello di machine learning principale utilizzato è il **k-Nearest Neighbors (k-NN)**, un classificatore basato sulla similarità tra campioni. Tuttavia, il codice è strutturato in modo flessibile, consentendo di integrare e testare altri algoritmi.  

### **Personalizzazione e Analisi dei Risultati**  
Il programma offre un elevato grado di personalizzazione, permettendo agli utenti di:  
- Scegliere la strategia di validazione  
- Configurare il pre-processing dei dati  
- Analizzare le prestazioni del modello attraverso metriche dettagliate  

I risultati vengono presentati in due modalità:  
1. **Esportazione dei dati** in formato CSV per un'analisi approfondita  
2. **Visualizzazione grafica**, con Confusion Matrix e Curva ROC per un’interpretazione immediata delle prestazioni del modello  

Grazie a queste funzionalità, il progetto si pone come un valido strumento di supporto per lo studio e la diagnosi dei tumori al seno, sfruttando il machine learning per migliorare la classificazione delle cellule tumorali. 