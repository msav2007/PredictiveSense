# Detection dataset audit

| class | images | boxes | zero-box images |
|---|---:|---:|---:|
| mug/cup | 437 | 717 | 0 |
| watch | 187 | 220 | 0 |

Total zero-box (hard-negative/background) images: **11**
Boxes below min area / extreme aspect: **0**

## Box area fraction (per class)
| class | min | median | p95 | max |
|---|---:|---:|---:|---:|
| mug/cup | 0.01014 | 0.14449 | 0.72773 | 0.99844 |
| watch | 0.0103 | 0.32772 | 0.95019 | 0.99875 |

## Blur variance (per class, higher = sharper)
| class | min | median | p95 | max |
|---|---:|---:|---:|---:|
| mug/cup | 2.60764 | 126.51638 | 1174.75403 | 4855.45678 |
| watch | 8.8636 | 496.15445 | 3169.7589 | 5876.19684 |

Within-class duplicate groups: 0 images across 0 groups
Across-class duplicate groups: 1
Exact byte-duplicate groups: 0
Duplicates against data/eval: 0

Licence breakdown: {'https://creativecommons.org/licenses/by/2.0/': 635}
