# dagmm
Auto-encodeur profond intégrant un modèle de mélange gaussien pour la détection non supervisée de fraudes

Ce repo consiste en mon projet de session pour le cours GLO-7030: Apprentissage par reseaux de neurones profonds à l'université Laval. Le code implémente un "Deep Autoencoding Gaussian Mixture Model (DAGMM)" pour faire de la détection d'anomalies de manière non supervisée sur un jeu de données dont les deux classes sont hautement débalancées (0.1727% d'anomalies). Plusieurs expériences ont été testées pour déterminer les facteurs influençant les performances. La meilleure configuration a obtenu, sur la classe "fraude", une précision de 78.93% et un rappel de 80.69% (pour un f-score de 79.80%) en 400 époques d'entraînement. Sur un GPU Nvidia GTX 1080, cela correspond à moins de 38 minutes.

Le jeu de données n'est pas inclus dans le repo en raison de sa taille. Sans lui, le code ne s'exécutera évidemment pas. Il est disponible sur Kaggle en suivant le lien suivant: 

https://www.kaggle.com/mlg-ulb/creditcardfraud

Pour que le code fonctionne alors sans problème, il suffit de prendre le fichier du jeu de données nommé "creditcard.csv" et de le placer dans le même dossier que les fichiers ".py".

Veuillez vous référer à l'article "Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection" de Zong et al. pour la théorie et les détails sur les DAGMM. Il est librement disponible au lien suivant: https://openreview.net/forum?id=BJJLHbb0-

Le fichier de code "dagmm.py" correspond à l'implémentation de la classe du DAGMM.

Le fichier "model.py" implémente la logique de préparation des données, d'entraînement et de test.

Le fichier "data_loader.py" permet de séparer le jeu de données en ensembles entraînement/test.

Le fichier "history.py" permet d'enregistrer et d'afficher les métriques d'entraînement.

Le fichier "utils.py" implémente des fonction utilitaires.

Le fichier "main.py" gère la configuration et l'exécution des expériences.

Le notebook jupyter "data_analysis.ipynb" correspond à une brève analyse des données.

Le notebook jupyter "unsupervised_benchmark.ipynb" implémente quelques méthodes classiques d'apprentissage non supervisé sur le même jeu de données pour agir à titre de comparatif.
