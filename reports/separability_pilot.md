# Separability pilot — offense / dual / defense in embedding space

Balanced sample (per bucket): {'dual': 1500, 'defense': 1500, 'offense': 1500}; each unit truncated to 2000 chars so length is not the signal. **Data-separability only** — not capability separability (that needs the unlearning-tax experiment).

### tfidf

- **offense vs defense probe:** accuracy 0.943, macro-F1 0.943 (chance 0.50)
- **3-way (offense/dual/defense) probe:** accuracy 0.934, macro-F1 0.934 (chance 0.33); labels ['defense', 'dual', 'offense'], confusion [[1341, 74, 85], [23, 1468, 9], [51, 53, 1396]]
- **centroid cosine distances:** offense–defense 0.628, offense–dual 0.576, defense–dual 0.572
- **dual substrate sits closer to `defense`**; it lies *between* offense and defense (both bucket–dual distances < offense–defense).

### semantic(bge-small)

- **offense vs defense probe:** accuracy 0.927, macro-F1 0.927 (chance 0.50)
- **3-way (offense/dual/defense) probe:** accuracy 0.900, macro-F1 0.900 (chance 0.33); labels ['defense', 'dual', 'offense'], confusion [[1276, 119, 105], [62, 1402, 36], [54, 74, 1372]]
- **centroid cosine distances:** offense–defense 0.067, offense–dual 0.060, defense–dual 0.063
- **dual substrate sits closer to `offense`**; it lies *between* offense and defense (both bucket–dual distances < offense–defense).
