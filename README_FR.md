# MODAL Order Book

Outils de simulation et d'estimation d'événements rares pour un projet de cours MODAL sur la dynamique des carnets d'ordres à cours limité. Ce projet est réalisé par **Lizhan Hong** et **Tom Zhang** sous la supervision du Professeur **Charles-Albert Lehalle**.
Le projet modélise la dynamique des files d'attente avec des files naissance-mort indépendantes et une excitation de type Hawkes, puis compare le Monte Carlo naïf par thinning d'Ogata et le Markovian Conditional Restart Splitting.

1. Monte Carlo naïf par thinning d'Ogata ;
2. **Markovian Conditional Restart Splitting**.

La base de code actuelle se concentre sur les modèles utilisés dans les notebooks finaux :

- `nb1_123_files_poisson_independantes.ipynb`
- `nb2_124_files_hawkes_couplees.ipynb`
- `nb3_125_deuxieme_limite.ipynb`


## Structure du dépôt

```text
MODAL_ORDER_BOOK/
├── model/
│   ├── __init__.py
│   ├── events.py
│   ├── lob.py
│   ├── Hawkes.py            # Wrapper de compatibilité pour les anciens imports en majuscules
│   ├── hawkes_4q.py         # Modèle Hawkes à quatre files pour les études de seconde limite
│   ├── hitting_times.py     # Temps d'atteinte naissance-mort de Poisson indépendants
│   ├── ogata.py             # Simulateurs avec reprise et trajectoires
│   ├── rare_events.py       # Problèmes d'événements rares, cibles et fonctions score
│   ├── restart_splitting.py # Markovian Conditional Restart Splitting
│   ├── splitting.py         # Fixed-Level Splitting et expériences AMS (legacy)
│   ├── analysis.py          # Utilitaires de comparaison d'estimateurs
│   └── utils.py             # RNG, chronométrage et utilitaires numériques
├── tests/
│   ├── test_restart_splitting.py
│   └── test_splitting_validation.py
├── nb1_123_files_poisson_independantes.ipynb
├── nb2_124_files_hawkes_couplees.ipynb
├── nb3_125_deuxieme_limite.ipynb
└── images/
```

Le fichier `.pytest_cache/README.md` est généré par pytest et ne fait pas partie de la documentation du projet. Le README du projet est ce fichier `README.md` à la racine.

## Concepts principaux

### Thinning d'Ogata

Le thinning d'Ogata est utilisé pour simuler des trajectoires de processus ponctuels pour la dynamique d'un carnet d'ordres pilotée par un processus de Hawkes. Il génère des événements candidats à partir d'une intensité majorante et accepte ou rejette chaque candidat selon l'intensité réelle courante.

Dans ce projet, Ogata est le simulateur de trajectoires de référence. Il n'est pas en lui-même un estimateur d'événements rares.

### Markovian Conditional Restart Splitting

Le **Markovian Conditional Restart Splitting** est la méthode d'événements rares demandée dans le cadre du projet. Le terme « splitting » ne désigne pas la décomposition d'une intensité Hawkes en horloges de référence et d'excitation.

Pour les modèles de Hawkes exponentiels, l'état augmenté `(N, S)` est Markovien, où `N` est le vecteur de files et `S` est l'état d'excitation à décroissance exponentielle courante. La méthode :

- collecte des checkpoints proches de la frontière, tels que `Q-1 = 1` ou `Q+1 = 1` ;
- estime les lois empiriques telles que `Law(S | Q-1 = 1)` ;
- redémarre des simulations locales à partir de checkpoints Markoviens complets échantillonnés ;
- estime les transitions rares locales, par exemple la déplétion avant récupération.

Le thinning d'Ogata reste le simulateur de trajectoires de bas niveau.

### Utilitaires de splitting legacy

`model/splitting.py` contient encore les utilitaires de Fixed-Level Splitting et d'AMS issus d'expériences antérieures. Ils restent compatibles à l'import, mais ne constituent pas la méthode demandée par le professeur pour le projet final.

Le Fixed-Level Splitting utilise des niveaux score choisis manuellement, par exemple :

```python
levels = [0.20, 0.40, 0.60, 0.80, 1.00]
```

À chaque niveau, les trajectoires ayant atteint le niveau survivent ; celles qui ne l'ont pas atteint sont éliminées. Les survivants sont rééchantillonnés à partir de leurs premiers checkpoints d'atteinte de niveau, puis poursuivis indépendamment. L'estimation de probabilité est le produit des probabilités de survie conditionnelles à travers les niveaux.

L'AMS choisit les niveaux automatiquement. Il simule une population de particules, élimine répétitivement la fraction la moins bien classée, clone à partir des meilleures particules au niveau adaptatif, et continue les clones avec des flux aléatoires indépendants.

Un seul run d'AMS fournit une estimation de probabilité mais pas d'erreur standard fiable. Pour des barres d'erreur de qualité publication, il faut effectuer plusieurs macro-réplications AMS indépendantes et calculer l'écart-type empirique.

### Checkpointing et mémoire Hawkes

Le checkpointing est essentiel. Un processus de Hawkes ne peut pas être redémarré à partir des seules tailles de files, car les intensités courantes dépendent des événements passés via l'état d'excitation.

L'objet checkpoint commun stocke :

- le temps courant
- l'état des files
- l'état d'excitation Hawkes
- le vecteur d'intensité courant
- le score courant
- les métadonnées nécessaires au problème d'événements rares

C'est ce qui rend le Markovian Conditional Restart Splitting valide pour les modèles de Hawkes.

### Notation à quatre files pour la seconde limite

Dans `nb3_125_deuxieme_limite.ipynb`, le code utilise l'ordre d'état suivant :

```text
[Q+1, Q-1, Q+2, Q-2]
```

Le notebook adopte l'interprétation suivante :

- indice positif : côté ask
- indice négatif : côté bid
- `Q-2 | Q-1 = 0` : seconde limite bid du même côté qu'une déplétion de première limite bid
- `Q-2 | Q+1 = 0` : seconde limite bid observée après la déplétion de la première limite opposée
- `Q-2_same = Q-2 | Q-1 = 0`
- `Q-2_opp = Q-2 | Q+1 = 0`

Pour le modèle Hawkes à quatre files, le vecteur d'excitation exposé au notebook est :

```text
S = [S^{+1,-}, S^{-1,-}, S^{+1,- -> +2,+}, S^{-1,- -> -2,+}]
```

Dans l'implémentation actuelle, il est extrait du checkpoint comme suit :

```text
[H[0], H[1], G[0], G[1]]
```

Ici `G[0]` est l'excitation croisée côté ask des retraits de `Q+1` vers les ajouts à `Q+2`, et `G[1]` est l'excitation croisée côté bid des retraits de `Q-1` vers les ajouts à `Q-2`. Aucune composante n'est dupliquée.

### Ce que visualise nb3

Les figures de la section 1.2.5 dans `nb3_125_deuxieme_limite.ipynb` sont générées à partir des redémarrages du Markovian Conditional Restart Splitting, et non à partir des anciennes expériences de Fixed-Level Splitting ou d'AMS.

Le notebook comprend :

- des résumés conditionnels empiriques d'excitation pour `Law(S | Q-1 = 1)` et `Law(S | Q+1 = 1)` ;
- des histogrammes marginaux du vecteur `S` à quatre composantes ;
- des distributions jointes bidimensionnelles des composantes importantes de `S` ;
- des tableaux de contenu des redémarrages montrant les états frontières échantillonnés, les excitations de début et de fin, les indicateurs de succès, les temps d'atteinte locaux et le `Q-2` final ;
- des histogrammes de `Q-2_same`, `Q-2_opp`, `Q-2 | Q-1 = 0` et `Q-2 | Q+1 = 0` ;
- des graphiques de sensibilité en fonction du paramètre d'excitation croisée `a_cross` ;
- des intervalles de confiance bootstrap pour les moyennes conditionnelles de `Q-2` et les différences same-minus-opposite ;
- un estimateur MCRS multiniveau réel sur les niveaux de files `[8, 6, 4, 2, 1, 0]` ;
- une comparaison à budget égal avec le Monte Carlo Ogata naïf.

Le thinning d'Ogata est toujours utilisé en interne dans le simulateur pour générer chaque trajectoire locale, mais l'estimateur et les quantités de la section 1.2.5 représentées sont basés sur le MCRS.

## API publique

Les imports les plus utiles sont :

```python
from model import (
    IndependentPoissonSimulator,
    SingleHawkesSimulator,
    CoupledHawkesSimulator,
    FourQueueHawkesSimulator,
    RareEventProblem,
    first_limit_depletion_problem,
    q1_depletion_problem,
    min_best_depletion_problem,
    second_limit_activation_problem,
    MarkovState,
    BoundarySample,
    RestartSplittingResult,
    collect_boundary_states,
    default_hawkes_burn_in,
    restart_from_boundary_distribution,
    run_naive_depletion_monte_carlo,
    multilevel_markovian_restart_splitting,
    bootstrap_mean_ci,
    bootstrap_difference_ci,
    summarize_conditional_S,
    run_markovian_conditional_restart_splitting,
)
```

Utilitaires de comparaison :

```python
from model.analysis import (
    run_markovian_conditional_restart_splitting,
    run_naive_boundary_mc_comparison,
    compare_restart_results,
    extract_q_neg2_restart_observables,
)
```

Les anciens imports Hawkes restent disponibles :

```python
from model.Hawkes import simulate_hawkes_queue, simulate_coupled_hawkes
from model.hawkes import simulate_hawkes_queue, simulate_coupled_hawkes
```

## Exemple rapide

```python
from model import CoupledHawkesSimulator
from model.analysis import (
    run_markovian_conditional_restart_splitting,
    run_naive_boundary_mc_comparison,
    compare_restart_results,
)

simulator = CoupledHawkesSimulator(
    mu_plus=1.5,
    mu_minus=1.1,
    alpha=0.15,
    beta=0.5,
    sign_convention="v4",
)

initial_state = [25, 25]

naive = run_naive_boundary_mc_comparison(
    simulator=simulator,
    initial_state=initial_state,
    queue_index=-1,
    horizon=80.0,
    n_paths=800,
    horizon_local=10.0,
    seed=42,
)

mcrs = run_markovian_conditional_restart_splitting(
    simulator=simulator,
    initial_state=initial_state,
    queue_index=-1,
    horizon=80.0,
    n_boundary_paths=800,
    horizon_local=10.0,
    n_restarts=2_000,
    seed=43,
)

table = compare_restart_results([naive, mcrs])
display(table)
```

## Colonnes du tableau de comparaison

Les notebooks présentent des tableaux avec les colonnes suivantes :

- `method` : nom de l'estimateur.
- `probability` : probabilité de déplétion locale avant récupération estimée.
- `std_error` : erreur standard estimée, lorsque disponible.
- `relative_error` : `std_error / probability`.
- `cpu_seconds` : temps d'exécution réel.
- `n_boundary_paths` : trajectoires complètes simulées pour collecter les états proches de la frontière.
- `n_boundary_samples` : checkpoints utiles collectés à la frontière.
- `n_restarts` : simulations locales redémarrées à partir d'états frontières empiriques.
- `n_successes` : redémarrages ayant atteint zéro avant la récupération.
- `n_events` : événements simulés acceptés.
- `n_candidates` : événements candidats Ogata, y compris les candidats rejetés.
- `cost_normalized_rel_error` : `relative_error * sqrt(cpu_seconds)`, mesure d'efficacité ajustée au coût approximative. Plus la valeur est faible, mieux c'est.

## Utilisation recommandée des notebooks

- `nb1` : vérification de cohérence Poisson. Il n'y a pas d'état d'excitation, donc `S` est vide et le Markovian Conditional Restart Splitting se réduit à un redémarrage depuis un état frontière de file.
- `nb2` : files Hawkes couplées. Utiliser le thinning d'Ogata avec la même politique de burn-in pour le Monte Carlo naïf et le Markovian Conditional Restart Splitting, puis montrer pourquoi la préservation de `S` est importante.
- `nb3` : analyse de seconde limite à quatre files. Utiliser le Markovian Conditional Restart Splitting pour `Q-2_same`, `Q-2_opp`, `Q-2 | Q-1=0` et `Q-2 | Q+1=0`. Le notebook montre également la distribution jointe empirique de `S` à la frontière et les échantillons de redémarrage réels utilisés pour produire les figures de seconde limite.

## Exécution des notebooks

Depuis la racine du dépôt :

```bash
jupyter notebook
```

Puis ouvrir :

```text
nb1_123_files_poisson_independantes.ipynb
nb2_124_files_hawkes_couplees.ipynb
nb3_125_deuxieme_limite.ipynb
```

Pour exécuter depuis la ligne de commande sans écraser les notebooks originaux :

```bash
jupyter nbconvert --to notebook --execute nb1_123_files_poisson_independantes.ipynb \
  --output /tmp/nb1_executed.ipynb --ExecutePreprocessor.timeout=900

jupyter nbconvert --to notebook --execute nb2_124_files_hawkes_couplees.ipynb \
  --output /tmp/nb2_executed.ipynb --ExecutePreprocessor.timeout=1200

jupyter nbconvert --to notebook --execute nb3_125_deuxieme_limite.ipynb \
  --output /tmp/nb3_executed.ipynb --ExecutePreprocessor.timeout=1800
```

## Tests

Lancer :

```bash
pytest -q
```

La suite de tests vérifie :

- la validation de queue Poisson contre une probabilité exacte
- le comportement legacy du Fixed-Level Splitting et de l'AMS
- le comportement frontière/redémarrage du Markovian Conditional Restart Splitting
- l'extraction conditionnelle de `S`, y compris le mapping à quatre composantes pour quatre files
- la reproductibilité déterministe avec des seeds fixes
- la préservation de la mémoire des checkpoints Hawkes
- les intensités Hawkes non-négatives
- l'excitation croisée à quatre files
- les utilitaires de convention de signe de nb3 pour `Q-2_same` et `Q-2_opp`
- les tests de fumée d'import publics

## Dépendances

Le code utilise :

- Python 3.12+
- NumPy
- pandas
- matplotlib
- SciPy, utilisé par les formules existantes dans les notebooks
- pytest, pour les tests
- Jupyter / nbconvert, pour l'exécution des notebooks

Aucun framework de simulation lourd n'est requis.

## Notes sur la reproductibilité

- Tous les estimateurs d'événements rares utilisent des flux `np.random.Generator` explicites.
- Les redémarrages du Markovian Conditional Restart Splitting reçoivent des flux RNG enfants indépendants.
- Les checkpoints Hawkes préservent la mémoire d'excitation.
- Les exemples des notebooks fixent des seeds pour des sorties reproductibles.

## Statut

L'implémentation actuelle est destinée à des expériences propres dans le cadre d'un projet de cours, et non à un usage en trading ou en calibration en production. Le workflow final compare le Monte Carlo Ogata naïf avec le Markovian Conditional Restart Splitting sous les mêmes paramètres de modèle et définitions de frontière. Les notebooks du projet utilisent désormais le nom de méthode exact **Markovian Conditional Restart Splitting** ; le Fixed-Level Splitting et l'AMS ne subsistent qu'en tant qu'utilitaires expérimentaux legacy.
