# Solver export interchange

Poker Coach Lab accepts two versioned interchange formats:

- `bundle-json-v1`: nested JSON with one object per solved private-hand node.
- `tabular-csv-v1`: one CSV row per available action, designed as a simple target for vendor-specific conversion scripts.

Both formats pass through the same `SolverBundleImporter`; CSV therefore has the same card, state, frequency, EV, provenance, duplicate-node and suit-isomorphism validation as JSON.

## Tabular CSV v1

The header requires these columns:

| Column | Meaning |
|---|---|
| `schema_version` | Generic bundle schema; currently `1.0.0`. |
| `source` | Solver or converter identity. |
| `source_version` | Exact solver/export/converter version. |
| `node_id` | Stable identifier shared by every action row at one decision node. |
| `game` | Canonical game id, currently `holdem_no_limit`. |
| `players` | Active players at the decision. |
| `hero_position` | Canonical hero position such as `BTN` or `BB`. |
| `effective_stack_bb` | Effective stack before the action, in big blinds. |
| `pot_bb` | Pot before the action, in big blinds. |
| `board` | Space-separated cards; empty preflop, then 3/4/5 cards. |
| `hero_cards` | Two space-separated private cards. |
| `action_history` | `|`-separated canonical prior actions. |
| `rake_model` | Stable name for the solved rake configuration. |
| `utility_model` | Usually `chip_ev`; tournament adapters may use an explicit payout/ICM model id. |
| `allowed_sizes` | `|`-separated normalized sizes included in the abstraction. |
| `action` | Canonical action id: `fold`, `check`, `call`, `bet:FRACTION`, or `raise_to:BB`. |
| `frequency` | Conditional action probability from 0 to 1. Frequencies at each node must sum to one. |
| `ev` | Action EV in big blinds under the declared utility model. |

All rows must retain one bundle-level `schema_version`, `source`, and `source_version`. Rows sharing `node_id` must have identical state columns; only `action`, `frequency`, and `ev` may change. Extra columns are retained by the source file but ignored by this adapter.

Literal suit names are canonicalized only for fingerprints and matching. The imported cards remain untouched for display and provenance. Globally renamed suits share one cache node, while different flush and blocker structures remain different nodes.

See [sample_solver_export.csv](C:/Users/st_ma/Documents/Negreanu/coach/examples/sample_solver_export.csv) for a four-street example.

## Import commands

Auto-detect JSON or CSV by extension/content:

```powershell
$env:PYTHONPATH = (Resolve-Path '.\src')
python -m poker_coach.solver_cli '.\examples\sample_solver_export.csv' --database '.\poker_coach_lab.sqlite3'
```

Force the adapter when an export uses a nonstandard extension:

```powershell
python -m poker_coach.solver_cli '.\export.txt' --format tabular-csv-v1 --database '.\poker_coach_lab.sqlite3'
```

The Solver Lab browser view exposes the same format selector, example loader, validator, and idempotent import path.

## Multi-street traversal

Imported node IDs survive SQLite round trips. `SolutionForest` links the nearest unambiguous ancestor when source/version, game, position, utility, rake, private ranks, action-history prefix, board prefix and suit-isomorphic card relationships agree. Stack size may decrease and active player count may stay constant or decrease.

Branches remain separate children. If two candidate parents are equally advanced or incomparable, the node remains a root with `ambiguous_parents`; the coach never invents an edge. The local API exposes the full forest at `GET /api/solution-tree` and a root-to-node path at `GET /api/solution-tree/{fingerprint}/path`.
