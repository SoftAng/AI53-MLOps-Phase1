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
Run ID (Run B): 19c50be4db0d4fd7bf8adb6e40748ffb

### 6.4 Verification Results (Run A & Run B)

* **Run A (`learning_rate: 0.001`) Run ID :** `f8e01a5e404b4f05baae19fdcd6b55e1` (`best_val_macro_f1 = 0.625419`, `val_accuracy = 0.6466`, `val_loss = 0.8471`)
* **Run B (`learning_rate: 0.0003`) Run ID :** `19c50be4db0d4fd7bf8adb6e40748ffb` (`best_val_macro_f1 = 0.6091`, `val_accuracy = 0.6165`, `val_loss = 0.8927`)

| Check / Change | VALIDATE | PREPARE | TRAIN | EVALUATE | Terminal Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Check 1: No change | REUSED | REUSED | REUSED | NOT RUN | `Stage 'validate' didn't change, skipping` / `Stage 'prepare' didn't change, skipping` / `Stage 'train' didn't change, skipping` |
| Check 2: Comment in `src/validate_data.py` | RERUN | REUSED | REUSED | NOT RUN | `Running stage 'validate'` / `Stage 'prepare' didn't change, skipping` / `Stage 'train' didn't change, skipping` |
| Run B: `train.learning_rate = 0.0003` | REUSED | REUSED | RERUN | NOT RUN | `'data\raw.dvc' didn't change, skipping` / `Stage 'validate' didn't change, skipping` / `Stage 'prepare' didn't change, skipping` / `Running stage 'train'` |

* **Comparaison avec les prédictions du 6.1 :** Les observations réelles correspondent exactement à nos prédictions : l'ajout d'un commentaire dans `src/validate_data.py` n'a relancé que `validate` grâce au rapport JSON déterministe, et la baisse du `learning_rate` n'a relancé que `train` en réutilisant les images préparées (`prepare`). Pour `EVALUATE`, DVC affiche `NOT RUN` plutôt que `REUSED` uniquement parce que la commande ciblait `dvc repro train` (qui s'arrête avant la branche d'évaluation). Par ailleurs, comme anticipé au 6.3, le `learning_rate = 0.0003` converge plus lentement sur 4 époques (`val_macro_f1 = 0.6091` contre `0.6254` pour Run A).


## Milestone 7: Compare Three Controlled Runs

### 7.1 Run C Hypothesis (`augmentation.enabled: true`)
* **Modification contrôlée par rapport au Run A :** `learning_rate` remis à `0.001` (valeur de base), et activation de l'augmentation de données (`augmentation.enabled: false` $\rightarrow$ `true`, avec `horizontal_flip: 0.5` et `rotation_degrees: 15`).
* **Effet attendu et justification :** L'augmentation de données applique des transformations aléatoires (miroir horizontal, rotations) à chaque époque, ce qui réduit le surapprentissage sur le petit jeu d'entraînement (1 034 images) et favorise la généralisation. Cependant, comme l'entraînement ne dure que 4 époques, la tâche d'apprentissage devient plus difficile à court terme : on s'attend à une perte d'entraînement (`train_loss`) plus élevée et à un score de validation (`val_macro_f1`) proche ou légèrement inférieur à celui du Run A sur 4 époques.


### 7.2 Validation Results Comparison

| Run | Controlled change | Val. Accuracy (Epoch 4) | Val. Macro-F1 (Epoch 4) | Training time (s) | MLflow Run ID |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Baseline (`lr=0.001, aug=False, dropout=0.30`) | `0.6391` | `0.6254` | `14.6s` | `f8e01a5e404b4f05baae19fdcd6b55e1` |
| **B** | Lower learning rate (`lr=0.0003`) | `0.6165` | `0.6091` | `13.2s` | `19c50be4db0d4fd7bf8adb6e40748ffb` |
| **C** | Data augmentation (`augmentation.enabled=True`, `rotation_degrees=10`) | `0.6541` | `0.6416` | `12.0s` | `fabfdbf39ed74d03893722308ab2ad55` |

* **Comparaison B vs A :** La baisse du taux d'apprentissage à `0.0003` réduit le `val_macro_f1` final de `0.6254` à `0.6091` (`-0.0163`). Cela confirme notre prédiction : sur seulement 4 époques, un pas d'apprentissage plus petit ralentit la convergence et ne permet pas au modèle d'atteindre le niveau de performance de la référence.
* **Comparaison C vs A :** L'activation de l'augmentation de données (`augmentation.enabled=True`) fait passer le `val_macro_f1` final de `0.6254` à `0.6416` (`+0.0162`) et la `val_accuracy` de `0.6391` à `0.6541`, tout en présentant une perte d'entraînement (`train_loss = 0.8769` vs `0.8672`) légèrement supérieure comme prévu. L'augmentation limite le surapprentissage et améliore la généralisation sur le jeu de validation. Toutefois, ces expériences reposant sur une seule graine aléatoire (`random_state=42`) et un petit jeu de validation de 133 images, cet écart modéré doit être interprété avec prudence et ne garantit pas à lui seul un gain statistique sur d'autres graines.

#### Engineering Checkpoint 6
* **Justification de la métrique principale (`val_macro_f1`) :** Pour une classification multiclasse à 3 catégories (`angular_leaf_spot`, `bean_rust`, `healthy`), la *Macro-F1* calcule la moyenne arithmétique du score F1 (équilibre précision/rappel) de chaque classe indépendamment de sa fréquence. Elle garantit ainsi que le modèle sélectionné détecte correctement les trois états sanitaires sans négliger une classe au profit des autres.
* **Pourquoi n'utiliser le jeu de test qu'après la sélection du modèle ?** Le jeu de test (`test`, 128 images) doit rester une mesure impartiale de la capacité de généralisation sur des données totalement inédites. Si on l'utilisait pour comparer A, B et C et choisir les hyperparamètres, on introduirait une fuite d'information (*data leakage* / surapprentissage sur le jeu de test), ce qui rendrait l'évaluation finale artificiellement optimiste.


## Milestone 8: Select and Test One Model

### 8.1 Sélection du modèle

La métrique principale de sélection est `val_macro_f1`, conformément au critère défini au Milestone 6. Les scores comparés sont ceux de la dernière époque, car c’est le modèle final de chaque Run qui est enregistré dans MLflow.

| Run | Run ID | `val_macro_f1` final |
| :--- | :--- | :---: |
| A | `f8e01a5e404b4f05baae19fdcd6b55e1` | `0.625419` |
| B | `19c50be4db0d4fd7bf8adb6e40748ffb` | `0.609117` |
| C | `fabfdbf39ed74d03893722308ab2ad55` | `0.641604` |

**Run sélectionné : Run C**, ID `fabfdbf39ed74d03893722308ab2ad55`. Il obtient le meilleur `val_macro_f1` final parmi les trois candidats. La règle de départage par `val_loss` n’a pas été nécessaire, car les scores principaux ne sont pas à égalité. Run C présente également une `val_accuracy` finale de `0.6541`.

Cette sélection s’appuie exclusivement sur les résultats de validation. Le jeu de test n’a pas été utilisé pour comparer ou choisir les modèles.

**Limite :** les Runs ont été comparés avec une seule graine aléatoire (`random_state=42`) et un jeu de validation de taille limitée. Le meilleur score observé ne garantit donc pas que Run C sera systématiquement supérieur avec d’autres graines ou sur d’autres données.


### 8.2 Résultats de l’évaluation finale

L’étape `evaluate` a chargé le modèle du Run C, ID `fabfdbf39ed74d03893722308ab2ad55`. Le champ `selected_run_id` de `reports/test_metrics.json` confirme qu’il s’agit bien du Run choisi.

| Run sélectionné | Test accuracy | Test Macro-F1 | Identité des données d’entraînement |
| :--- | :---: | :---: | :--- |
| `fabfdbf39ed74d03893722308ab2ad55` | `0.640625` | `0.636634` | `92f093303c75f5af` |

Le modèle a correctement classé 82 images sur 128. Ces résultats évaluent le modèle sélectionné sur le jeu de test, qui n’a pas servi à la sélection. Ils doivent être interprétés avec prudence, car ce jeu contient un nombre limité d’images.


