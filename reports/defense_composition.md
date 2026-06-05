# Defense corpus composition by source

Source: `defensive_documents.parquet` — **162 documents** → **2902 analysis units** (resegmented at ~3000 chars). Classified by URL. The unit column is what actually enters the experiments; NIST SP 800-series standards are large and dominate units while being few documents.

| category | docs | % docs | units | % units | median src chars |
|---|---:|---:|---:|---:|---:|
| NIST_SP_800 | 5 | 3% | 1741 | 60% | 941372 |
| OTHER | 12 | 7% | 326 | 11% | 14275 |
| US_GOV_CISA | 7 | 4% | 233 | 8% | 31127 |
| ACADEMIC | 9 | 6% | 179 | 6% | 51888 |
| SUPPLEMENT_GITHUB | 101 | 62% | 146 | 5% | 1676 |
| MITRE_CWE | 3 | 2% | 114 | 4% | 16560 |
| VENDOR | 20 | 12% | 94 | 3% | 5768 |
| IETF_RFC | 5 | 3% | 69 | 2% | 35268 |

## Headline

- **NIST SP 800-series: 5/162 documents (3%) but 1741/2902 units (60%).**
- Compliance/policy-flavored sources (NIST SP + US-gov/CISA): **68% of units**.
- Median source length by category: NIST_SP_800 941372c, OTHER 14275c, US_GOV_CISA 31127c, ACADEMIC 51888c, SUPPLEMENT_GITHUB 1676c, MITRE_CWE 16560c, VENDOR 5768c, IETF_RFC 35268c.

## Verdict

The defense unit corpus is **compliance/policy-dominated**: a few large NIST SP 800-series standards (plus US-gov/CISA) supply the majority of units. The contribution claim should be framed as *defensive-governance/compliance* content, not broad operational defensive mechanism knowledge — or the corpus should be rebalanced toward mechanism-level defensive sources before strong claims.
