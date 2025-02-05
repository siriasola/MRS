# **AI per la Classificazione dei Tumori al Seno**  

Il tumore al seno può essere suddiviso in due categorie principali: **benigno** e **maligno**. Le cellule benigne tendono a rimanere localizzate, con una crescita controllata e non invasiva, mentre quelle maligne presentano un comportamento aggressivo, caratterizzato da una rapida proliferazione e dalla capacità di invadere i tessuti circostanti e formare metastasi. La corretta classificazione delle cellule tumorali è essenziale per una diagnosi precoce e per la scelta del trattamento più adeguato.  

La distinzione tra cellule benigne e maligne avviene attraverso l’analisi di diverse caratteristiche morfologiche e biologiche, come la forma e la dimensione delle cellule, la struttura del nucleo, il grado di adesione e il tasso di mitosi. Questi parametri vengono estratti da test citologici e rappresentano informazioni fondamentali per determinare la natura del tumore.  

### **Obiettivo del Progetto**  
Questo progetto si propone di sviluppare un modello di **machine learning** capace di classificare le cellule tumorali in **benigne** o **maligne** con elevata accuratezza. A tal fine, verrà utilizzato il dataset **Breast Cancer Wisconsin (Original)**, ampiamente impiegato nella ricerca per la classificazione dei tumori al seno. L’algoritmo implementato potrà essere utilizzato come supporto ai medici per migliorare la precisione e la rapidità della diagnosi.  

### **Metodologia**  
Per garantire una valutazione affidabile del modello, verranno applicate diverse tecniche di validazione, tra cui:  
- **Holdout** 
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

### Come Eseguire il Codice ###
1. **Clonare o Scaricare il Repository**  
   Scaricare l’intero progetto, contenente le cartelle `data`, `models`, `preprocessing`, `tests`, `validation` e il file `main.py`.

2. **Installare le Dipendenze**  
   Assicurarsi di avere installato Python (versione 3.7 o superiore) e installare le librerie necessarie elencate nel file `requirements.txt` (se presente). In caso contrario, le librerie di base utilizzate sono:  
   ```bash
   pip install pandas numpy matplotlib
   ```
3. **Eseguire lo Script Principale** 
    Dal terminale, posizionarsi nella cartella radice del progetto e lanciare il comando:
    ```
    python main.py 
    ```
    o, su sistemi Unix/macOS: 
    ```
    python3 main.py
    ```
4. **Interagire con il Programma** 
    Il programma è interattivo: chiederà all’utente di specificare:
    - Il percorso del dataset (default: ```data\version_1.csv ```)
    - Il metodo di gestione dei valori mancanti (media, moda, mediana)
    - Il metodo di scaling delle feature 
    - La strategia di validazione (Holdout, K-Fold Cross Validation, Leave-One-Out Cross Validation)
    - Il valore *k* per il k-NN
    Al termine, verranno mostrati i risultati a schermo e verranno generate eventuali visualizzazioni (Confusion Matrix, Curva ROC).

### Formati Supportati ###
 Il progetto è stato pensato per essere generalizzabile a diversi formati di file, oltre a ```CSV```. In particolare: 
 - CSV (```.csv```)
 - Excel (```.xlsx, .xls```)
 - JSON (```.json```)
 - TSV (```.tsv```)
 - TXT (testo tabellato)
 Nel file ```preprocessing/importdata.py``` viene determinato il tipo di file in base all’estensione del percorso specificato e viene eseguito il caricamento appropriato usando pandas.

 ### Struttura del Progetto ###
 La struttura delle cartelle e dei file principali è la seguente:
├── data
│   └── version_1.csv
│
├── models
│   ├── __init__.py
│   └── k_nearest_neighbor.py
│
├── preprocessing
│   ├── __init__.py
│   ├── data_cleaner.py
│   ├── importdata.py
│   └── normalizzazione.py
│
├── tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_evaluation.py
│   └── test_knn.py
│
├── validation
│   ├── __init__.py
│   ├── datasplit.py
│   ├── evaluation.py
│   ├── metriche.py
│   └── visualizzazione.py
│
└── main.py
## Descrizione delle Cartelle ##
- **data/**: Contiene il file version_1.csv (dataset di esempio).
- **models/**: Contiene i modelli di Machine Learning (in questo caso k_nearest_neighbor.py).
- **preprocessing/**: Contiene gli script per caricare i dati e applicare operazioni di pulizia, gestione dei valori mancanti e scaling.
- **tests/**: Contiene i test automatici per la validazione delle funzionalità del codice.
- **validation/**: Contiene gli script per la suddivisione del dataset (holdout, k-fold, leave-one-out), il calcolo delle metriche e la visualizzazione dei risultati.
- **main.py**: File principale che coordina l’esecuzione di tutti i moduli e fornisce un’interfaccia interattiva all’utente.

## **1. Caricamento del DataSet**  
Nel file ```preprocessing/importdata.py``` è definita la classe ```DatasetProcessor``` che, attraverso il metodo ```load_data()```, rileva automaticamente l’estensione del file e carica i dati in un oggetto ```pandas.DataFrame```.

# Flusso di Caricamento: #

1. L’utente digita (o conferma) il percorso del file quando richiesto dal programma.
2. In base all’estensione (```.csv```, ```.xls```, ```.xlsx```, ```.json```, ```.tsv```, ```.txt```), il metodo ```load_data()``` applica la funzione di lettura corrispondente di ```pandas```.
3. Se il caricamento ha successo, viene restituito un ```DataFrame```, altrimenti ```None```.

## **2. Pulizia del DataSet**  
Nel file ```preprocessing/data_cleaner.py``` troviamo due classi principali:

1. ```DataCleaner```
   - Rimuove i duplicati con ```df.drop_duplicates()```.
   - Rimuove le righe che non hanno il valore di target, specificato tramite il parametro ```target_column```.
2. ```MissingValueHandler```
   - Gestisce i **valori mancanti** all’interno delle colonne numeriche.
   - L’utente può scegliere fra 3 strategie: **mean**, **median**, **mode**.
   - Se la scelta non è valida, viene usato di default il metodo della media.
Queste operazioni assicurano che il dataset sia privo di duplicati e che le righe senza target vengano rimosse, per evitare inconsistenze nella fase di addestramento. Successivamente, i valori mancanti nelle feature numeriche vengono riempiti con la strategia scelta, in modo da ridurre l’impatto di dati incompleti.

## **3. Configurazione Interattiva** 
Dopo aver caricato il dataset, l’utente può interagire con il programma per specificare le tecniche di **pulizia** e **scaling**.

# **3.1 Gestione dei Valori Mancanti**  
All’interno del flusso di ```main.py```, la funzione ```handle_missing_values(df)``` invoca:
- ```choose_missing_value_method()```: chiede all’utente di scegliere il metodo fra ```mean```, ```median``` e ```mode```.
- Crea quindi un oggetto ```MissingValueHandler``` con il metodo prescelto.
- Applica ```MissingValueHandler.clean(df)``` per riempire i valori mancanti.

# **3.2 Scaling delle Feature** 
Nel file ```preprocessing/normalizzazione.py``` è presente la classe ```FeatureScaler```. Le opzioni di scaling offerte sono:
- **Normalizzazione**: Trasforma le feature in un range [0, 1].
- **Standardizzazione**: Trasforma le feature in modo che abbiano media = 0 e deviazione standard = 1.
Anche in questo caso, ```main.py``` fornisce una funzione (```scale_features(df)```) che chiede all’utente di scegliere la tecnica desiderata e poi applica la trasformazione corrispondente.

## **4. Classificazione: k-NN** 
Il **k-Nearest Neighbors (k-NN)** è un algoritmo di Machine Learning basato sulla similarità dei campioni.

Nel file ```models/k_nearest_neighbor.py``` è definita una classe (```ClassificatoreKNN```) che contiene:

- ```__init__(self, k=5)```: Il costruttore, dove k è il numero di vicini da considerare.
- ```train(self, features, labels)```: Salva internamente i dati di training, senza vera fase di addestramento (k-NN è un metodo basato sulla memoria).
- ```Euclidian_distance(self, point)```: Calcola la distanza euclidea tra point e tutti i campioni di training.
- ```k_nearest_neighbor(self, point)```: Trova i k punti più vicini a quello specificato.
- ```predict(self, point)```: Restituisce la classe del punto in input.
- ```predict_batch(self, points)```: Applica la previsione a un insieme di punti (batch).

## **5. Validazione del Modello** 
All’interno della cartella ```validation/```:
1. ```datasplit.py```
   - Suddivide il dataset in **training** e **test** (metodo ```split_holdout(train_size)```).
   - Esegue la **K-Fold Cross Validation** suddividendo i dati in k fold.
   - Esegue la **Leave-One-Out Cross Validation (LOO-CV)**, in cui a ogni iterazione viene utilizzato 1 singolo campione come test.
2. ```evaluation.py```
   - Coordina l’intero processo di validazione, richiamando le funzioni di split e i calcoli delle metriche.
   - Metodi principali:
      - ```valutazione_holdout(self, train_size=0.8)```
      - ```valutazione_k_fold(self)```
      - ```valutazione_leave_one_out(self)```
L’utente, durante l’esecuzione di ```main.py```, sceglie la strategia di validazione preferita (Holdout, K-Fold, Leave-One-Out). In base a questa scelta, vengono chiamati i metodi corrispondenti di datasplit ed evaluation.

## **6. Metriche** 
Nel file ```validation/metriche.py``` troviamo la classe (```MetricheCrossValidation```) e le funzioni correlate, che calcolano le seguenti metriche:
- **Accuracy Rate**
- **Error Rate**
- **Sensitivity** 
- **Specificity**
- **Geometric Mean**
- **Area Under the Curve**

*Oltre ai metodi di calcolo, troviamo funzioni per:*

*Validazione Holdout*
*K-Fold Cross Validation*
*Leave-One-Out Cross Validation*
*e metodi per plot e salvataggio CSV delle metriche aggregate.*

## **7. Visualizzazione e Salvataggio dei Risultati** 


## **8. Conclusioni** 
Questo progetto fornisce una base solida per affrontare la classificazione di tumori al seno (o di altre tipologie di dataset con struttura simile) utilizzando il k-NN. La pipeline copre l’intero flusso di lavoro:

1. Caricamento dei dati da formati diversi.
2. Pulizia (rimozione duplicati, gestione valori mancanti).
3. Scaling delle feature.
4. Scelta della strategia di validazione (Holdout, K-Fold, Leave-One-Out).
5. Addestramento e Predizione con k-NN.
6. Calcolo delle Metriche (Accuracy, Sensitivity, Specificity, ecc.).
7.Visualizzazione (Confusion Matrix, Curva ROC) e Salvataggio dei risultati.
