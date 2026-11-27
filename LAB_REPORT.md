# AI53 | MLOps Project — Phase 1 Lab Report

## Milestone 0: Repository Audit

### 1. Problèmes identifiés dans le dépôt initial

| # | Fichier / Composant | Problème identifié | Impact MLOps |
|---|---|---|---|
| 1 | `src/reproducibility.py` | `set_reproducibility()` ne fixe que `random.seed()` (les TODO pour NumPy, PyTorch CPU/CUDA et cuDNN sont vides). De plus, `seed_worker()` est vide (`pass`) et `make_generator()` retourne un générateur non initialisé. | Les entraînements et le mélange des lots (*DataLoaders*) ne sont pas reproductibles d'une exécution à l'autre. |
| 2 | `src/validate_data.py` | La fonction `validate_dataset()` lève une `NotImplementedError`. | Aucune porte de contrôle qualité ne vérifie l'intégrité des images, les classes ou le nombre de fichiers avant l'entraînement. |
| 3 | `src/train.py` (et `provenance.py`) | Les blocs `TODO 1` à `TODO 7` sont vides : le manifeste de provenance n'est pas généré avant l'entraînement, et ni les paramètres, ni les métriques par époque, ni les artefacts, ni le modèle PyTorch ne sont enregistrés dans MLflow. | Perte totale de traçabilité des expériences, des hyperparamètres et des versions de code/données associées à un modèle. |
| 4 | Racine du projet & `.gitignore` | `.gitignore` exclut déjà `data/raw/`, `data/processed/` et `models/*.pt`, mais DVC n'est pas encore initialisé et aucun fichier `dvc.yaml` n'existe. | Les données brutes ne sont pas versionnées et l'exécution des étapes (`validate`, `prepare`, `train`, `evaluate`) n'est pas automatisée. |
| 5 | `params.yaml` | Le champ `selection.run_id` est vide (`""`). | Aucun modèle candidat n'est encore sélectionné pour l'évaluation finale sur le jeu de test. |
| 6 | Environnement & Git | Le dépôt Git n'est pas encore initialisé avec un historique de commits ni lié à un dépôt distant GitHub. | Impossible de suivre l'évolution du code ou de capturer un SHA Git valide pour la provenance des exécutions. |

### 2. Engineering Checkpoints (Milestone 0)
* **Pourquoi `.gitignore` ignore-t-il `data/raw/` et `models/*.pt` ?**
  Ces dossiers contiennent des fichiers volumineux et binaires (images et poids de réseaux de neurones). Les ignorer dans Git évite d'alourdir le dépôt ; ils seront suivis séparément via DVC (fichiers `.dvc` et cache DVC).
* **Pourquoi la reproductibilité nécessite-t-elle plus que `random.seed(seed)` ?**
  Un pipeline PyTorch utilise plusieurs générateurs aléatoires indépendants (Python standard, NumPy, PyTorch CPU, PyTorch GPU/CUDA, ainsi que les workers parallèles des `DataLoader`). Tous doivent être fixés et les opérations non déterministes de cuDNN doivent être désactivées.


## Milestone 1: Repeatability & Random Seeding

### Engineering Checkpoint 2
* **Pourquoi fixer uniquement `random.seed(seed)` ne suffit-il pas dans un pipeline PyTorch ?**
  PyTorch, NumPy et Python utilisent chacun leur propre générateur de nombres pseudo-aléatoires (PRNG) indépendant. De plus, sur GPU (CUDA/cuDNN), le choix automatique d'algorithmes rapides (`cudnn.benchmark = True`) introduit du non-déterminisme matériel, et les processus parallèles (`workers`) d'un `DataLoader` doivent recevoir leur propre graine dérivée (`seed_worker`) ainsi qu'un `torch.Generator` dédié (`make_generator`) pour garantir un mélange (`shuffle`) et une augmentation de données reproductibles.
* **Quelles limites subsistent malgré ces réglages ?**
  La reproductibilité exacte au bit près reste sensible aux changements de matériel (différents modèles de GPU ou exécution CPU vs GPU), aux changements de version de CUDA/PyTorch, ainsi qu'à certaines opérations atomiques parallèles sur GPU qui ne disposent pas d'implémentation déterministe.


## Milestone 2: Data Versioning with DVC

### Engineering Checkpoint 3
* **Comment DVC et Git collaborent-ils pour versionner `data/raw` ?**
  Les images brutes sont placées dans `data/raw/` (ignoré par Git) et stockées dans le cache/remote DVC sous forme d'empreintes numériques (hash MD5). DVC génère un petit fichier métadonnée `data/raw.dvc` contenant ce hash, lequel est versionné dans Git. Chaque commit Git pointe ainsi vers une version exacte et immuable des données sans stocker de fichiers binaires lourds dans Git.
* **Que se passerait-il si une image était modifiée ou supprimée dans `data/raw` ?**
  `dvc status` détecterait immédiatement que le hash réel du dossier ne correspond plus à celui enregistré dans `data/raw.dvc`. On peut alors soit restaurer les données d'origine via `dvc checkout`, soit enregistrer la nouvelle version avec `dvc add data/raw` et un nouveau commit Git.


## Milestone 4: DVC Pipeline Design

### 4.2 Design & Dependencies Analysis
* **Should `PREPARE` depend on `reports/data_quality.json`?**
  Oui. Cela crée une dépendance explicite dans le DAG : `prepare` ne peut pas s'exécuter tant que la porte de contrôle qualité (`validate`) n'a pas réussi et généré ce rapport.
* **Should `PREPARE` also depend directly on `data/raw`?**
  Oui. `reports/data_quality.json` est un résumé statistique qui ne contient pas les pixels des images ; `prepare` doit lire directement les fichiers bruts dans `data/raw` pour les redimensionner et générer les manifestes.
* **Which changes in `params.yaml` should make `TRAIN` run again?**
  Uniquement les paramètres qui influencent l'entraînement et le suivi : `data.image_size`, `train.*`, `augmentation.*`, `compute.device` et `tracking.*`. Modifier `selection.run_id` ou `evaluation.batch_size` ne doit pas déclencher un réentraînement.
* **Why should `EVALUATE` track `selection.run_id`?**
  `evaluate` charge le modèle candidat depuis MLflow à l'aide de cet identifiant. Suivre `selection.run_id` garantit que DVC rejoue `evaluate` uniquement lorsqu'un nouveau modèle candidat est sélectionné pour le test final.
* **Which file contains scores that DVC should track as metrics?**
  `reports/train_summary.json` (pour l'étape `train`) et `reports/test_metrics.json` (pour l'étape `evaluate`) contiennent les métriques numériques et doivent être déclarés sous `metrics:` avec `cache: false`.


## Milestone 5: Record What Each MLflow Run Used

### Engineering Checkpoint 4
* **Git :** Enregistre la version exacte du code source (le hash SHA du commit et l'indicateur *dirty* signalant d'éventuelles modifications locales non commitées) utilisée pour exécuter l'entraînement.
* **DVC :** Enregistre les empreintes numériques exactes des jeux de données et des artefacts lourds (via `.dvc` et `dvc.lock`) ainsi que les dépendances du pipeline qui les ont produits.
* **MLflow :** Enregistre le journal complet de chaque exécution d'entraînement, incluant ses hyperparamètres, le jeu de données d'entrée, les métriques à chaque époque, les artefacts générés et le modèle PyTorch entraîné avec sa signature.


## Milestone 6: Predict Which Stages Need to Run Again

### 6.1 Predictions Before Running DVC

| Change | VALIDATE | PREPARE | TRAIN | EVALUATE |
| :--- | :--- | :--- | :--- | :--- |
| Change `train.learning_rate` | REUSED | REUSED | RERUN | REUSED |
| Change `data.image_size` | REUSED | RERUN | RERUN | NEEDS UPDATE |
| Edit `src/validate_data.py`; the quality report stays the same | RERUN | REUSED | REUSED | REUSED |
| Edit only `README.md` | REUSED | REUSED | REUSED | REUSED |

#### Engineering Checkpoint 5
* **Changement de paramètre (`data.image_size`) :** `VALIDATE` est `REUSED` car il ne lit pas `params.yaml`. `PREPARE` est `RERUN` car `data.image_size` figure dans ses `params:` ; il réécrit alors `data/processed/*` et les manifestes. `TRAIN` est `RERUN` car il suit `data.image_size` et dépend de `data/processed/train` et `validation`. `EVALUATE` est `NEEDS UPDATE` car il suit `data.image_size` et dépend de `data/processed/test` dont le contenu a changé.
* **Modification de fichier (`src/validate_data.py`, rapport inchangé) :** `VALIDATE` est `RERUN` car `src/validate_data.py` est déclaré dans ses `deps:`. Cependant, comme `reports/data_quality.json` est déterministe (sans horodatage), son hash reste identique après exécution : `PREPARE`, `TRAIN` et `EVALUATE` conservent exactement les mêmes entrées et sont donc `REUSED`.


### 6.3 Pre-Training Decisions
* **Métrique principale et critère de départage :** La métrique principale de sélection est **`val_macro_f1`** afin d'accorder le même poids à chacune des 3 classes. En cas d'égalité (à 4 décimales), le départage se fera sur la **`val_loss`** la plus faible (puis `val_accuracy` la plus élevée).
* **Effet attendu d'une baisse du `learning_rate` (`0.001` $\rightarrow$ `0.0003`) :** Avec un pas d'apprentissage plus petit sur seulement 4 époques, la descente de gradient sera plus régulière et moins oscillante, mais la convergence risque d'être plus lente (pouvant donner une perte de validation plus stable ou un F1 légèrement inférieur si 4 époques ne suffisent pas à converger).


Run ID (Run A): f8e01a5e404b4f05baae19fdcd6b55e1