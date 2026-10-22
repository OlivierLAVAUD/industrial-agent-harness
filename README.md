# Industrial Agent Harness

**PoC public : supervision humaine prouvable d'un agent GenAI de maintenance industrielle, avec traçabilité vérifiable (tamper-evident) et conformité AI Act (Art. 12 et 14).**

Ce dépôt démontre une chaîne complète, exécutable et testable :

1. un agent LangGraph interroge un graphe de connaissance Neo4j en **GraphRAG multi-sauts** ;
2. il propose une action de maintenance, puis demande son exécution ;
3. le harness intercepte l'action, applique une politique et **bloque l'exécution tant qu'un humain n'a pas approuvé** ;
4. chaque appel, proposition, demande, décision et résultat est inscrit dans un **registre d'audit chaîné par HMAC** ;
5. l'opérateur approuve ou rejette depuis une console **Streamlit** exécutée dans le même processus que l'agent.

Le scénario de référence est : **« L'équipement P-204 présente une vibration anormale. Que faire ? »**

---

## 1. Problème résolu

Un agent de maintenance peut recommander une intervention critique, mais une recommandation n'est pas une autorisation d'exécution. Pour être déployable dans un environnement industriel régulé, le système doit prouver :

- **qui** a demandé quoi, **quand**, **sur quel équipement**, **à partir de quelles sources** et **sous quelle politique** ;
- **qui** a approuvé ou refusé, et **pour quelle raison** ;
- que l'action n'a pas été exécutée sans approbation ;
- que les traces n'ont pas été modifiées après coup.

Le PoC répond à ces exigences avec deux propriétés centrales :

| Propriété | Mécanisme |
|---|---|
| **Supervision humaine effective** | Politique `require_approval` sur `execute_maintenance_action`, décision humaine bloquante, exécution conditionnelle |
| **Traçabilité vérifiable** | Journal append-only JSONL, HMAC-SHA256 par événement, chaînage `prev_hash`, vérification intégrale |

### Correspondance AI Act

| Exigence AI Act | Mécanisme implémenté | Preuve observable |
|---|---|---|
| **Art. 12 — Journalisation automatique** | `harness/audit.py` : chaque `tool_call`, `approval_requested`, `approval_decided`, `tool_result`, `tool_blocked` et cycle d'agent est journalisé avec horodatage UTC, acteur, outil et payload | `verify_audit_chain()` retourne `{"valid": true, "events_checked": N}` ; l'UI affiche la vérification |
| **Art. 12 — Traçabilité des données et du raisonnement** | Contexte GraphRAG conservé dans la demande d'approbation : chemin `Équipement → Symptôme → Historique → Procédure`, sources Neo4j, capteurs | Panneau d'approbation Streamlit et événement `approval_requested` |
| **Art. 14 — Supervision humaine** | `policy.yaml` impose une approbation pour l'exécution ; l'outil métier n'est appelé qu'après décision `approved` | Boutons Approuver/Rejeter, blocage `ACTION_BLOQUEE`, refus tracé |
| **Art. 14 — Capacité d'arrêter / d'ignorer une action** | Le rejet empêche l'appel à l'outil d'exécution ; l'agent reçoit explicitement le refus | `tool_blocked` + réponse finale « action non autorisée » |
| **Art. 13 — Transparence et explicabilité** | Le chemin GraphRAG, les procédures, les historiques, les capteurs et les sources sont affichés à l'approbateur | Trace complète de l'agent et panneau d'approbation |
| **Art. 15 — Robustesse et cybersécurité** | Chaîne HMAC, clé hors code, vérification à chaque rendu, échec explicite en cas de modification | Tests de falsification et de suppression d'enregistrements |

> Ce PoC démontre des **mécanismes techniques**. Il ne constitue pas une certification AI Act : l'analyse de conformité, la documentation technique, la gestion des risques et l'évaluation humaine restent à produire.

---

## 2. Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                        Console Streamlit (ui/app.py)                  │
│  Lance l'agent · affiche la trace · approuve/rejette · vérifie audit  │
└───────────────┬───────────────────────────────────────┬──────────────┘
                │ même processus                        │ décision
                ▼                                       ▼
┌──────────────────────────────┐          ┌────────────────────────────┐
│   Agent LangGraph            │          │   Harness de gouvernance   │
│   agent/graph.py             │          │   harness/policy.py        │
│   agent (LLM + bind_tools)   │          │   harness/approval.py      │
│        ↕ ToolNode            │          │   harness/audit.py         │
│   agent/tools.py             │          │   harness/governance.py    │
└───────────────┬──────────────┘          └──────────────┬─────────────┘
                │ outils enveloppés                        │
                ▼                                          ▼
┌──────────────────────────────┐          ┌────────────────────────────┐
│   Connaissance GraphRAG      │          │   Registre d'audit HMAC    │
│   graph/queries.py           │          │   JSONL append-only        │
│   Neo4j 5.26                 │          │   vérification de chaîne   │
│   30 nœuds / 27 relations    │          │                            │
└──────────────────────────────┘          └────────────────────────────┘
```

### Les trois couches

#### 1. Couche connaissance — `graph/`

- **Neo4j 5.26** stocke un graphe métier volontairement petit mais réaliste : 6 équipements, 6 symptômes, 6 procédures, 6 historiques, 6 capteurs (**30 nœuds**) et **27 relations**.
- **GraphRAG plutôt que RAG vectoriel** : la question exige un raisonnement explicite multi-sauts et des sources identifiables. Le graphe rend le chemin `P-204 → vibration anormale → historique 2024-03-12 → REMPLACER_ROULEMENT` lisible et vérifiable. Un index vectoriel retrouverait des passages proches, mais pas la structure causale ni la provenance exacte.
- `graph/seed_graph.py` crée les contraintes d'unicité, les index (`name`, `description`, `code`, `date`) et les données avec `MERGE`. Le script est idempotent et supporte `--reset`.
- `graph/queries.py` exécute une requête Cypher multi-sauts, retourne des dictionnaires structurés, formate le chemin et produit le contexte injecté dans l'agent.

#### 2. Couche agent — `agent/`

- **LangGraph** construit une boucle explicite `agent → tools → agent → … → END`.
- Le nœud `agent` utilise un modèle de chat avec `bind_tools` ; le nœud `tools` est un `ToolNode` LangGraph. **Le function calling natif du LLM est donc réellement utilisé.**
- Trois outils seulement :
  - `query_knowledge_graph` : lecture GraphRAG, auto-autorisée ;
  - `propose_maintenance_action` : proposition, auto-autorisée, sans exécution ;
  - `execute_maintenance_action` : exécution simulée, **toujours soumise à approbation**.
- Aucun outil ne contient de logique de politique ou d'approbation : l'enveloppe de gouvernance est appliquée par `harness/governance.py`.
- Si `OPENAI_API_KEY` est absente, un **modèle scripté** reproduit le protocole `AIMessage(tool_calls=...)`. Le graphe, le `ToolNode` et la gouvernance sont donc exercés à l'identique hors-ligne. Ce n'est pas un faux chemin de code : seul le modèle est remplacé.

#### 3. Couche gouvernance — `harness/`

- `policy.py` tente d'utiliser `preloop.PolicyManager` puis retombe sur un adaptateur YAML explicite (`LocalPolicyManager`).
- `approval.py` tente d'utiliser `preloop.ApprovalManager` puis retombe sur un magasin en mémoire thread-safe (`LocalApprovalManager`) : création de la demande avec contexte complet, attente bloquante, décision, expiration.
- `audit.py` tente d'utiliser `preloop.AuditLogger` puis retombe sur `LocalAuditLogger`, journal HMAC-SHA256 chaîné.
- `governance.py` enveloppe chaque outil. Si un serveur Preloop est configuré (`PRELOOP_URL`, `PRELOOP_TOKEN`), `langchain_preloop.wrap_with_preloop` (alias `PreloopTool`) est utilisé. Sinon, le fallback local applique la politique, demande l'approbation, journalise et exécute conditionnellement.

### Pourquoi ces choix

| Choix | Justification |
|---|---|
| **Neo4j** | Le domaine est relationnel et explicable ; Cypher exprime naturellement les chemins multi-sauts et les contraintes d'intégrité. |
| **GraphRAG vs RAG vectoriel** | Traçabilité de la provenance, chemins causaux lisibles, pas d'ambiguïté d'embedding sur un petit domaine critique. |
| **LangGraph** | Boucle agent/outil explicite, état typé, testabilité, intégration naturelle avec `ToolNode` et le function calling. |
| **Preloop** | Séparation entre logique métier et contrôle : politique, approbation hors bande, journal d'audit. L'intégration réelle passe par `langchain-preloop`. |
| **MCP** | Contrat d'outils standardisé, réutilisable par d'autres clients que Streamlit ; démontré dans `scripts/mcp_server.py`. |
| **Streamlit** | Console de supervision minimale, même processus que l'agent, aucun broker externe pour le PoC. |
| **Docker Compose** | Reproductibilité Neo4j + seed + application, et exécution des tests dans le conteneur. |

### Intégration Preloop : réelle, optionnelle, et fallback explicite

Deux distributions existent :

- `langchain-preloop` est l'intégration LangChain/Preloop. Le harness l'utilise réellement dès que `PRELOOP_URL` et `PRELOOP_TOKEN` sont renseignés, via `wrap_with_preloop`.
- `preloop` sur PyPI est le **serveur auto-hébergé** (FastAPI, PostgreSQL, etc.), pas un SDK exposant `PolicyManager`, `ApprovalManager` ou `AuditLogger`. Ces classes ne sont pas disponibles dans la version installée.

Conformément à la consigne de fallback, le PoC fournit donc des implémentations locales **clairement identifiées** dans `harness/`. Elles ne réimplémentent pas le moteur CEL de Preloop : elles couvrent uniquement les primitives nécessaires à la démonstration (lecture YAML, store d'approbation en mémoire, journal HMAC). En mode Preloop réel, les approbations sont gérées par Preloop hors de la console Streamlit ; la console reste la démonstration du flux local.

Le serveur Preloop n'est pas inclus dans l'image Docker (dépendances lourdes : PostgreSQL/pgvector, Kubernetes, etc.). Il peut être installé séparément avec l'extra `preloop-server`.

---

## 3. Démonstration

### Prérequis

- Docker Desktop avec Docker Compose v2.24+ ;
- environ 2 Go d'espace libre ;
- une clé OpenAI **optionnelle**.

### Lancement

```bash
cp .env.example .env          # Windows : Copy-Item .env.example .env
# facultatif : renseigner OPENAI_API_KEY dans .env
docker compose up --build
```

Sans `OPENAI_API_KEY`, le modèle scripté prend le relais : la démonstration reste complète et déterministe.

Le service `seed` s'exécute une fois et doit se terminer avec :

```text
Seed terminé : 30 nœuds, 27 relations.
```

Puis ouvrir **http://localhost:8501**.

### Parcours guidé

1. Cliquer sur **« Lancer la démonstration »**.
2. Observer la trace : appel `query_knowledge_graph`, contexte GraphRAG, puis `propose_maintenance_action`.
3. L'agent appelle `execute_maintenance_action`. Le harness crée une demande d'approbation et **bloque l'exécution**.
4. Le panneau **« Approbation humaine requise »** affiche :
   - l'équipement et l'action (`P-204`, `REMPLACER_ROULEMENT`) ;
   - le chemin de raisonnement ;
   - les sources Neo4j consultées ;
   - la politique `require_approval` et l'identifiant de la demande.
5. Saisir l'identité de l'approbateur (simulée), puis :
   - **Approuver** : l'agent reprend, l'outil d'exécution simulée s'exécute, un ticket est produit ;
   - **Rejeter** : une raison est obligatoire ; l'exécution reste bloquée et l'agent explique le refus.
6. Vérifier en bas de page : **« Chaîne d'audit valide — N événement(s) vérifié(s) »**.
7. Déplier **« Derniers événements du registre d'audit »** pour inspecter les enregistrements HMAC.

### Vérification headless

```bash
# Dans le conteneur
docker compose exec app python scripts/demo.py --auto-approve

# Ou en local, après avoir exporté NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD
python scripts/demo.py --auto-approve
```

Le script affiche la trace, puis :

```text
Audit : valid=True events=10
```

Un rejet peut être testé avec `python scripts/demo.py --reject`.

### Exécution des tests

```bash
docker compose exec app pytest
```

En local, sans Neo4j, les tests d'intégration sont automatiquement ignorés :

```bash
pytest
```

---

## 4. Structure du dépôt

```text
industrial-agent-harness/
├── README.md
├── LICENSE
├── docker-compose.yml
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── policy.yaml
├── .env.example
├── graph/
│   ├── connection.py        # driver Neo4j
│   ├── queries.py           # Cypher multi-sauts + formatage
│   └── seed_graph.py        # 30 nœuds, 27 relations, contraintes, index
├── agent/
│   ├── graph.py             # LangGraph + function calling + modèle scripté
│   └── tools.py             # 3 outils métier sans logique de gouvernance
├── harness/
│   ├── policy.py            # PolicyManager réel ou fallback YAML
│   ├── approval.py          # ApprovalManager réel ou store local
│   ├── audit.py             # AuditLogger réel ou registre HMAC local
│   └── governance.py        # PreloopTool réel ou enveloppe locale
├── ui/
│   └── app.py               # console Streamlit
├── scripts/
│   ├── demo.py              # scénario CLI headless
│   └── mcp_server.py        # exposition MCP stdio des outils gouvernés
└── tests/
    ├── conftest.py
    ├── test_graph.py
    ├── test_harness.py
    ├── test_agent.py
    ├── test_audit_chain.py
    └── test_integration.py
```

---

## 5. Modèle de données et scénario

Le seed crée exactement :

| Label | Nombre | Exemples |
|---|---:|---|
| `Equipment` | 6 | `P-204` (pompe centrifuge), `C-101`, `M-310`, `V-450`, `E-120`, `R-077` |
| `Symptom` | 6 | `S-VIB-01` (vibration anormale), `S-TEMP-01`, `S-BRUIT-01`, `S-FUITE-01`, `S-PRESS-01`, `S-DEBIT-01` |
| `Procedure` | 6 | `PR-ROUL-01` / `REMPLACER_ROULEMENT`, `PR-ALIGN-01`, etc. |
| `MaintenanceHistory` | 6 | `H-2024-03-12` |
| `Sensor` | 6 | `CAP-VIB-01` (7,8 mm/s, statut alerte) |

Relations : `PRÉSENTE` (7), `RECOMMANDE` (8), `A_CONSTATÉ` (6), `MESURE` (6).

Le chemin retourné pour `P-204` + « vibration » est :

```text
P-204 (Pompe centrifuge) → vibration anormale → historique 2024-03-12
→ procédures REMPLACER_ROULEMENT, REALIGNER_ARBRE
```

Sources collectées : `P-204`, `S-VIB-01`, `PR-ROUL-01`, `PR-ALIGN-01`, `H-2024-03-12`, `CAP-VIB-01`.

---

## 6. Configuration

`.env.example` contient les variables attendues :

| Variable | Rôle |
|---|---|
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` | Connexion Neo4j |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | LLM réel (optionnel) |
| `APPROVER_ID` | Identité simulée de l'approbateur |
| `AUDIT_HMAC_KEY` | Clé HMAC du registre (à régénérer en production) |
| `AUDIT_LOG_PATH` | Chemin du registre JSONL |
| `APPROVAL_MODE` | `streamlit` (défaut), `cli`, `auto_approve`, `auto_reject` |
| `APPROVAL_TIMEOUT_SECONDS` | Délai avant expiration d'une demande |
| `PRELOOP_URL`, `PRELOOP_TOKEN` | Active l'enveloppe Preloop réelle |

Générer une clé HMAC :

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

`policy.yaml` est volontairement minimal :

```yaml
tools:
  query_knowledge_graph:
    policy: auto
  propose_maintenance_action:
    policy: auto
  execute_maintenance_action:
    policy: require_approval
```

---

## 7. MCP

`scripts/mcp_server.py` expose les trois outils gouvernés sur stdio :

```bash
# Démonstration non interactive
APPROVAL_MODE=auto_approve python scripts/mcp_server.py

# En conteneur
docker compose exec -e APPROVAL_MODE=auto_approve app python scripts/mcp_server.py
```

Le script tente `langchain_mcp_adapters.create_mcp_server` (nom prévu par l'architecture cible) puis retombe sur `mcp.server.fastmcp.FastMCP`, car cet helper n'est pas présent dans les versions publiées de `langchain-mcp-adapters`. Le contrat exposé reste identique : les trois outils enveloppés par la gouvernance.

---

## 8. Tests

La suite couvre :

- **`test_audit_chain.py`** : chaîne valide, altération de payload, suppression d'un enregistrement, mauvaise clé HMAC.
- **`test_harness.py`** : mapping des politiques, délégation au manager, attente Streamlit, décision, mode CLI.
- **`test_agent.py`** : ordre `query → propose → execute`, boucle LangGraph, approbation demandée pour l'exécution, blocage en cas de rejet.
- **`test_graph.py`** : formatage, extraction, comptage exact des nœuds/relations, chemins P-204, idempotence du seed.
- **`test_integration.py`** : scénario complet approuvé, scénario rejeté, boucle Streamlit avec approbation réelle via `AppTest`.

Résultat attendu dans l'environnement Docker :

```text
31 passed
```

---

## 9. Limites assumées

Ce PoC est volontairement borné et ne doit pas être présenté comme un système de production.

- **Domaine simulé** : 30 nœuds, 6 équipements, aucun connecteur GMAO/CMMS réel. L'exécution produit un ticket fictif.
- **Identité d'approbateur simulée** : aucun SSO, aucune authentification forte, aucun contrôle d'habilitation. En production, l'identité doit venir d'un IdP et la décision doit être signée.
- **Fallback Preloop local** : les classes `PolicyManager`, `ApprovalManager` et `AuditLogger` attendues par l'architecture cible ne sont pas exposées par la distribution PyPI `preloop` (serveur). Le PoC utilise donc le fallback local documenté pour la démonstration. L'intégration réelle `langchain-preloop` est disponible quand un serveur Preloop est configuré, mais la console Streamlit ne pilote alors pas les approbations Preloop.
- **Clé HMAC** : stockée dans une variable d'environnement. En production, utiliser un KMS/HSM, une rotation de clé et un stockage d'audit immuable (WORM). Le registre JSONL local n'est pas un WORM.
- **Troncature de la fin du journal** : la chaîne HMAC détecte une altération, une suppression au milieu ou un réordonnancement, mais pas la suppression des derniers enregistrements sans ancrage externe. En production, publier périodiquement la tête de chaîne (Merkle root, horodatage qualifié, WORM) pour rendre cette troncature détectable.
- **Store d'approbation en mémoire** : mono-processus, non partagé entre réplicas. Le PoC assume une seule instance Streamlit. Une montée en charge exigerait un store partagé et une reprise après incident.
- **Modèle scripté hors-ligne** : déterministe et destiné à la démonstration/aux tests. Avec un LLM réel, les réponses et le choix des procédures peuvent varier ; la gouvernance et l'audit restent identiques.
- **MCP** : le helper `create_mcp_server` de l'architecture cible n'existe pas dans les versions publiées ; un fallback FastMCP officiel est utilisé.
- **Conformité AI Act** : le PoC fournit des preuves techniques (journalisation, supervision, traçabilité), pas une certification ni une analyse de risque complète.

---

## 10. Ce que démontre le PoC

- Un agent peut raisonner en multi-sauts sur un graphe métier et **expliquer ses sources**.
- Une action sensible peut être **matériellement bloquée** tant qu'un humain n'a pas décidé.
- La décision humaine et le contexte de raisonnement sont **journalisés dans une chaîne vérifiable**.
- Une falsification, une suppression au milieu du journal ou une ré-écriture d'événement est **détectée** (la troncature de la fin nécessite un ancrage externe, voir limites).
- Le même flux est exécutable en **Streamlit**, en **CLI headless**, en **tests** et via **MCP**, sans changer la logique de gouvernance.

C'est la propriété centrale du harness : **la logique métier de l'agent est séparée de la logique de contrôle, et cette dernière laisse une preuve vérifiable de chaque décision.**

---

## 11. Licence

Apache License 2.0. Voir [LICENSE](LICENSE).
