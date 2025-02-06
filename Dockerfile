FROM python:3.10-slim

# Imposta la directory di lavoro
WORKDIR /app1

# Copia il file requirements.txt e installa le dipendenze
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia il dataset dentro il container
COPY Data/ /app1/data/

# Copia il resto dell'applicazione
COPY . ./

# Specifica il comando di avvio
CMD ["python", "main.py"]
