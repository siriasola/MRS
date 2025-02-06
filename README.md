# **AI per la Classificazione dei Tumori al Seno**  

Il tumore al seno può essere suddiviso in due categorie principali: **benigno** e **maligno**. Le cellule benigne tendono a rimanere localizzate, con una crescita controllata e non invasiva, mentre quelle maligne presentano un comportamento aggressivo, caratterizzato da una rapida proliferazione e dalla capacità di invadere i tessuti circostanti e formare metastasi. La corretta classificazione delle cellule tumorali è essenziale per una diagnosi precoce e per la scelta del trattamento più adeguato.  

La distinzione tra cellule benigne e maligne avviene attraverso l’analisi di diverse caratteristiche morfologiche e biologiche, come la forma e la dimensione delle cellule, la struttura del nucleo, il grado di adesione e il tasso di mitosi. Questi parametri vengono estratti da test citologici e rappresentano informazioni fondamentali per determinare la natura del tumore.  

### **Obiettivo del Progetto**  
Questo progetto si propone di sviluppare un modello di **machine learning** capace di classificare le cellule tumorali in **benigne** o **maligne** con elevata accuratezza. A tal fine, verrà utilizzato il dataset **Breast Cancer Wisconsin (Original)**, ampiamente impiegato nella ricerca per la classificazione dei tumori al seno. L’algoritmo implementato potrà essere utilizzato come supporto ai medici per migliorare la precisione e la rapidità della diagnosi.  

### **Analisi Dataset version_1.csv**

Il dataset version_1.csv svolge un ruolo centrale in questo progetto, fornendo i dati necessari per l’analisi e la classificazione dei tumori. Di seguito è riportata una descrizione dettagliata del suo contenuto:
- Numero di Campioni: Variabile a seconda della versione del dataset, in genere centinaia di campioni.   
- Numero di Caratteristiche: 13 caratteristiche per ciascun campione.
- Descrizione delle Caratteristiche:
- Blood Pressure: Pressione sanguigna registrata (non direttamente correlata alle cellule, possibile variabile aggiuntiva).
- Mitoses: Frequenza delle mitosi, indicativa del grado di proliferazione cellulare.
- Sample code number: Identificativo univoco per ogni campione di analisi (non utilizzato nell’analisi, ma mantenuto nel     dataset).
- Normal Nucleoli: Numero di nucleoli normali presenti nelle cellule.
- Single Epithelial Cell Size: Dimensione della singola cellula epiteliale, un indicatore della regolarità.
- Uniformity of Cell Size: Uniformità delle dimensioni cellulari; valori elevati possono indicare malignità.
- Clump Thickness: Spessore del gruppo di cellule, utile per valutare la densità dei campioni.
- Heart Rate: Frequenza cardiaca (aggiunto per scopi di studio, non direttamente legato alla classificazione).
- Marginal Adhesion: Capacità delle cellule di aderire tra loro.
- Bland Chromatin: Cromatina omogenea, legata all’aspetto dei nuclei cellulari.
- classtype_v1: Etichetta di classificazione delle cellule tumorali (2 = benigno, 4 = maligno).
- Uniformity of Cell Shape: Uniformità della forma delle cellule, importante per identificare alterazioni morfologiche.
- Bare Nucleix_wrong: Nuclei scoperti (probabilmente un errore di digitazione, riferito a “Bare Nuclei”).

## Personalizzazione e Analisi dei Risultati

Il programma offre un elevato grado di personalizzazione, permettendo agli utenti di:

- Configurare il pre-processing dei dati.
- Scegliere la strategia di validazione.
- Analizzare le prestazioni del modello attraverso metriche dettagliate.

### Modalità di Presentazione dei Risultati

I risultati vengono presentati in due modalità:

1. **Esportazione dei dati** in formato **Excel**, per un'analisi approfondita.
2. **Visualizzazione grafica**, con **Confusion Matrix** e **Curva ROC**, per un’interpretazione immediata delle prestazioni del modello.

Grazie a queste funzionalità, il progetto si pone come un valido strumento di supporto per lo studio e la diagnosi dei tumori al seno, sfruttando il **Machine Learning** per migliorare la classificazione delle cellule tumorali.

### Come Eseguire il Codice ##
1. **Installare le Dipendenze:**
Prima di eseguire il programma, è necessario assicurarsi di avere **Python 3.7** o superiore installato nel sistema e installare le dipendenze. Questo può essere fatto eseguendo il comando nella directory principale del progetto:

```bash
pip install -r requirements.txt
```
2. **Eseguire lo Script Principale:** 
    Dal terminale, spostarsi nella cartella del progetto ed eseguire:

    Windows
    ```
    python main.py 
    ```
    o, su sistemi Unix/macOS: 
    ```
    python3 main.py
    ```
3. **Interagire con il Programma:** 
Il programma è interattivo, chiederà all’utente di specificare:

- **Il percorso del dataset**
- **Il metodo di gestione dei valori mancanti** (Media, Moda, Mediana)
- **Il metodo di scaling delle feature** (Normalizzazione, Standardizzazione)
- **La strategia di validazione**  
  - Holdout  
  - K-Fold Cross Validation  
  - Leave-One-Out Cross Validation  
- **La selezione delle metriche di valutazione**  
  - Accuracy Rate  
  - Sensitivity  
  - AUC  
  - e altre metriche disponibili...  
- **Il valore di _k_ per il k-NN**

Al termine, verranno mostrati i risultati a schermo e verranno generate eventuali visualizzazioni, tra cui:

- **Confusion Matrix**
- **Curva ROC**

### Formati supportati ##

Il programma è stato progettato per analizzare dataset provenienti da diversi formati di file.  
Per garantire il corretto funzionamento, il file deve essere in uno dei seguenti formati:

- **.csv** (valori separati da virgola)
- **.xlsx** (Excel)
- **.json** (formato JSON)
- **.txt** (con delimitatore `,`)
- **.tsv** (con delimitatore `\t`)

Se il formato del file non è tra quelli supportati, verrà generato un errore e il programma interromperà l’elaborazione.

---

### 1. Caricamento del Dataset  

Durante l’esecuzione del programma, l’utente deve fornire il percorso del file contenente il dataset.  
Se il percorso non viene specificato o il file non è valido, il programma restituirà il seguente messaggio di errore:

> **"Errore nel caricamento del file:"**

Se il file viene caricato correttamente, il dataset sarà pronto per essere elaborato nelle fasi successive.

---

### 2. Pulizia del Dataset  

Dopo il caricamento dei dati, viene eseguita un’operazione di pulizia per garantire la qualità del dataset.  
Questa fase include:

1. **Rimozione dei duplicati**: vengono eliminati eventuali record duplicati nel dataset.  
2. **Gestione dei valori mancanti**: le righe prive di un valore nella colonna `classtype_v1` (la variabile target) vengono eliminate.  
3. **Riempimento dei valori mancanti**: l’utente può scegliere come trattare i dati mancanti nelle altre colonne, selezionando una delle seguenti opzioni:
   - **Media (Mean)**: sostituisce i valori mancanti con la media della colonna.  
   - **Mediana (Median)**: sostituisce i valori mancanti con la mediana della colonna (**opzione predefinita**).  
   - **Moda (Mode)**: sostituisce i valori mancanti con il valore più frequente.  

   Se l’utente non specifica una scelta valida, il programma utilizza automaticamente la **media**.

---

### 3. Scaling delle Feature  

Per migliorare le prestazioni degli algoritmi di apprendimento automatico, il dataset viene sottoposto a un processo di trasformazione delle feature numeriche.  
L’utente può scegliere tra due tecniche di scaling:

1. **Normalizzazione (Min-Max Scaling)**: ridimensiona i valori delle feature in un intervallo compreso tra **0 e 1**.  
2. **Standardizzazione (Z-score Scaling)**: trasforma i valori affinché abbiano **media 0 e deviazione standard 1**.  

Per garantire un’elaborazione corretta, alcune colonne del dataset **non** vengono modificate durante questa fase:

- **classtype_v1**: rappresenta la colonna target, che indica la classe del tumore (**benigno** o **maligno**).  
- **Sample code number**: è un identificativo univoco per ogni campione e **non** ha rilevanza nel processo di scaling.  

Se l’utente non specifica una scelta valida, il programma utilizzerà la **normalizzazione (Min-Max Scaling)** come impostazione predefinita.
## 4. Classificazione: k-NN

Il **k-Nearest Neighbors (k-NN)** è un algoritmo di Machine Learning supervisionato utilizzato per la classificazione dei dati in base alla somiglianza con i campioni già noti.  
Nel contesto di questo progetto, il k-NN viene impiegato per classificare i tumori come **benigni** o **maligni**, confrontando ogni nuovo campione con i dati presenti nel dataset di training.

L’algoritmo si basa su tre passaggi fondamentali:

1. **Calcolo della distanza** tra il nuovo campione e tutti i campioni presenti nel dataset.  
2. **Selezione dei k vicini più prossimi**, ovvero i campioni più simili in base alla distanza calcolata.  
3. **Assegnazione della classe** al nuovo campione in base alla classe predominante tra i vicini.

---

### 4.1 Funzionamento  

Il classificatore k-NN è stato implementato per gestire in modo efficiente il processo di classificazione.  
Le principali fasi dell’algoritmo sono le seguenti:

- **Fase di Addestramento:**  
  In questa fase, i dati vengono **memorizzati** all’interno del modello, senza eseguire una vera e propria fase di apprendimento.  
  I dati di training includono le feature (le caratteristiche dei campioni) e le rispettive etichette di classe.

- **Calcolo della Distanza:**  
  Per determinare la similarità tra i campioni, viene utilizzata una misura di distanza.  
  Nel progetto, la **distanza Euclidea** è il metodo scelto per confrontare i campioni, valutando quanto un punto sia vicino o lontano dagli altri nel dataset.

- **Identificazione dei k Vicini più Vicini:**  
  Dopo aver calcolato la distanza tra il nuovo campione e tutti i campioni di training, l’algoritmo seleziona i **k campioni più vicini**.  
  La scelta di **k** è un parametro determinante e può influenzare la qualità della classificazione.

- **Assegnazione della Classe:**  
  Il nuovo campione viene assegnato alla classe più frequente tra i suoi **k vicini**.  
  Se vi è un pareggio tra due o più classi, il sistema risolve la situazione scegliendo **casualmente** tra le classi con la stessa frequenza.

- **Predizione per un Singolo Campione o per più Campioni:**  
  Il classificatore permette di effettuare previsioni **sia su un singolo campione** sia **su più campioni contemporaneamente**.

- **Probabilità di Appartenenza a una Classe:**  
  Oltre a fornire la classe finale, il modello può calcolare la **probabilità** che un dato campione appartenga a una specifica classe.  
  Questa probabilità è determinata dalla proporzione di vicini appartenenti alla classe di interesse.

---

### 4.2 Gestione delle Situazioni  

- **Dati Mancanti:**  
  Il modello verifica la presenza di eventuali valori mancanti e **li rimuove** prima di effettuare la classificazione per evitare errori nei calcoli.

- **Scelta di k:**  
  Il valore di **k** deve essere scelto con attenzione:  
  - Un valore **troppo basso** può rendere il modello sensibile al rumore nei dati.  
  - Un valore **troppo alto** può portare a una classificazione troppo generalizzata.

- **Pareggio tra Classi:**  
  Se tra i **k vicini** ci sono più classi con la stessa frequenza, il modello **sceglie casualmente** tra le classi con il numero maggiore di occorrenze.

---

## 5. Validazione del Modello  

Per garantire l’affidabilità delle previsioni del modello k-NN, il progetto implementa diverse tecniche di validazione dei dati.  
L’obiettivo della validazione è **valutare le prestazioni del modello su dati non visti**, ridurre il rischio di **overfitting** e confrontare le metriche di valutazione in scenari differenti.

---

### 5.1 Suddivisione dei Dati  

Il file **datasplit.py** si occupa di suddividere il dataset in insiemi di **training** e **test**, utilizzando tre diverse strategie di validazione:

- **Holdout Validation:**  
  Il modello KNN viene addestrato con l’**80% dei dati** e testato sul restante **20%**.  
  Si ottiene un valore per ciascuna metrica di valutazione (**es. accuratezza, sensibilità, specificità**), che indica le prestazioni del modello.

- **K-Fold Cross Validation:**  
  Il modello viene **addestrato e testato k volte**, e alla fine si ottiene una **media** delle metriche di valutazione.  
  Questo fornisce una **stima più affidabile** della performance rispetto all’Holdout, poiché ogni dato viene usato **sia per il training che per il testing**.

- **Leave-One-Out Cross Validation (LOO-CV):**  
  Il modello viene testato una volta per **ogni campione**, il che fornisce una **valutazione estremamente accurata**, soprattutto su dataset di **piccole dimensioni**.  
  Tuttavia, il **costo computazionale** è molto elevato, poiché il modello viene addestrato **N volte**, dove **N è il numero totale di dati nel dataset**.

---

## 6. Metriche  

Durante la fase di validazione del modello, vengono misurate diverse metriche per valutare l’efficacia del classificatore k-NN.  
Questi indicatori permettono di analizzare la **precisione delle previsioni** e la capacità del modello di distinguere correttamente tra le diverse classi.

Le metriche vengono calcolate per ciascuna delle strategie di validazione adottate, fornendo una valutazione **completa** delle prestazioni del modello.

Di seguito sono riportate le metriche calcolate dal nostro sistema:

- **Accuracy Rate:** Percentuale di predizioni corrette sul totale dei campioni. Indica la precisione generale del modello. **Valore ideale: vicino a 1**.  
- **Error Rate:** Percentuale di predizioni errate sul totale. Più basso è il valore, migliore è la performance del modello. **Valore ideale: vicino a 0**.  
- **Sensitivity:** Misura la capacità del modello di individuare correttamente i **casi positivi** (**tumori maligni**). **Valore ideale: vicino a 1**.  
- **Specificity:** Misura la capacità del modello di identificare correttamente i **casi negativi** (**tumori benigni**). **Valore ideale: vicino a 1**.  
- **Geometric Mean:** Indica il bilanciamento tra **Sensitivity** e **Specificity**, utile in dataset sbilanciati. **Valore ideale: vicino a 1**.  
- **Area Under the Curve:** Rappresenta la capacità del modello di distinguere tra **classi positive e negative**.  
  Un valore vicino a **1** indica un’**elevata capacità discriminativa**.

---

## 7. Visualizzazione e Salvataggio dei Risultati  

Le metriche calcolate vengono archiviate in un file **Excel** denominato **validation_results.xlsx** all’interno della cartella **results/**.  
Questo permette di consultare facilmente le prestazioni del modello e confrontare diversi esperimenti.

Oltre ai dati numerici, il programma genera **visualizzazioni** che aiutano a interpretare meglio i risultati del modello.  
Tra questi troviamo:

- **Matrice di Confusione:** Un’analisi dettagliata degli errori di classificazione.  
- **Curva ROC:** Mostra le performance del modello in termini di **sensibilità e specificità**.  
- **Grafico a Barre delle Metriche:** Un confronto visivo tra le metriche di valutazione per evidenziare i punti di forza e debolezza del modello.

Tutti i grafici vengono salvati come immagini nella cartella **results/** e successivamente **integrati nel file Excel**, ognuno in un **foglio separato**, per facilitare la consultazione.

---

## 8. Conclusioni  

Questo progetto fornisce una base solida per affrontare la classificazione di **tumori al seno** (o di altre tipologie di dataset con struttura simile) utilizzando il **k-NN**.  
La pipeline copre l’intero flusso di lavoro:

- **Caricamento dei dati** da formati diversi.  
- **Pulizia** (rimozione duplicati, gestione valori mancanti).  
- **Scaling delle feature**.  
- **Scelta della strategia di validazione** (Holdout, K-Fold, Leave-One-Out).  
- **Addestramento e Predizione con k-NN**.  
- **Calcolo delle Metriche** (Accuracy, Sensitivity, Specificity, ecc.).  
- **Visualizzazione** (Confusion Matrix, Curva ROC).  
- **Salvataggio dei risultati**.