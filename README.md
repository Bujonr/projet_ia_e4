# projet_ia_e4
Ia - Deep learning visant à détecter les cyberattaques sur des systèmes type radar.

Utilisation de tensorflow 2.0

**run.py :** 

Permet de lancer l'ensemble de l'appli

**requirements.txt :** 

Liste l'ensemble des libraries nécessaires. 
Utilisable via la commande `python3 -m pip install -r requirements.txt̀`

**network folder :**

Contient l'ensemble des fichiers nécessaires à la récupération des données, leur traitement, leur visualisation et l'entraînement de l'IA

**network/__init__.py :**

Contient l'instance du réseau défini dans `network_model.py`, les données récupérées via le fichier `data_process.py` et permet la visualisation des résultats via les fonctions définies dans `viewdata.py`

**network/data_process.py :**

Récupère les données via les sondes définies avec la partie **cyber**, permet de récupérer les données en boucle telle que 0 = état normal, 1 = attaque et 100 = fin

**network/network_model.py :**

Définie la classe qui permettra d'instancier le réseau ainsi que les fonctions utiles à l'entraînement

**network/viewdata.py :**

Définie les fonctions utiles à la visualisation de l'évolution de l'accuracy et la perte (loss) sur les données d'entraînement et de validation (pas de cross validation mais validation classique)

