# Entanglement profile (Stage A)

Per offensive subcapability: how entangled its ATT&CK techniques are with D3FEND defenses, measured as distinct D3FEND counterparts per technique via the 14,003-row mappings (exact id ∪ parent rollup).

- Techniques assigned: **691** (orphans: 0)
- Dual-use contestedness threshold (p75 of mapped): **28.0** D3FEND counterparts
- Topic partition regions: defensive=271, dual-use=222, offensive=469

| subcapability | n_tech | mapped_frac | mean_contest | max | dual_topics | graph_blind |
|---|---|---|---|---|---|---|
| persistence_escalation | 145 | 0.97 | 25.46 | 50 | 51 |  |
| credential_access | 57 | 1.00 | 22.35 | 48 | 9 |  |
| defense_evasion | 146 | 0.83 | 17.86 | 40 | 48 |  |
| command_and_control | 41 | 0.98 | 14.66 | 29 | 6 |  |
| initial_access | 22 | 0.91 | 13.05 | 34 | 5 |  |
| execution | 46 | 0.65 | 12.89 | 30 | 6 |  |
| collection_exfiltration | 50 | 0.88 | 10.22 | 29 | 4 |  |
| discovery_lateral_movement | 59 | 0.73 | 7.22 | 31 | 1 |  |
| impact | 33 | 0.64 | 6.58 | 24 | 0 |  |
| reconnaissance | 92 | 0.00 | 0.00 | 0 | 92 | YES |

## Key finding: the reconnaissance paradox

`reconnaissance` (ATT&CK Reconnaissance + Resource-Development) has **zero D3FEND mappings** — the graph is structurally blind to pre-compromise activity — yet it is the canonical dual-use capability (scanning/enumeration/OSINT shared by red and blue teams). It is therefore flagged dual-use *by construction*, not by the graph. This is the central evidence that graph-contestedness alone cannot define dual-use.
