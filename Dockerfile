# Image de base : Python 3.11 léger (la diapo utilise 3.9, qui n'est plus maintenu)
FROM python:3.11-slim

WORKDIR /workspace

# 1) Dépendances d'abord : cette couche est mise en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Code du projet (voir .dockerignore pour ce qui est exclu)
COPY . .

# 3) Pas de dépôt Git dans l'image : DVC doit fonctionner sans Git
RUN dvc config core.no_scm true

# 4) Le conteneur exécute DVC ; par défaut, il lance le pipeline complet
ENTRYPOINT ["dvc"]
CMD ["repro"]