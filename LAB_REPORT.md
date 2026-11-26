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


