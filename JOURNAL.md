# JOURNAL
| ID | Date | Statut | Tag |
|---|---|---|---|
| P0a-B1 / R1 / V1 | 2026-10-04 | VALIDE | palier-0a |

## Récap palier 0a (2026-10-04)
- État : infra en place, package `bot` 0.0.1 (code dans src/), commit d8f1f11.
- Env : Python 3.14.7, .venv, versions figées dans requirements.lock (numpy 2.5.3, pandas 3.0.6, pytest 9.1.1).
- Tests : 3 passés (repo local + clone propre, revérifiés par le contrôleur).
- Limites connues : setuptools (build) non figé ; numpy/pandas non bornés dans pyproject (lock fait foi) ; README avec placeholder `<url-du-repo>`.
- Remote GitHub présent (BlumberMan/bot-trading-ia), rien poussé.
- Prochaine étape : palier 0b, valeurs GONOGO.md à faire valider par Iyad.
