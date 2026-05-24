# Empirical Analysis of Synthetic Samples

This document evaluates five synthetic IELTS Writing Task 2 samples
(`synthetic_samples.csv`) to assess target label alignment and their utility for
Automated Essay Scoring (AES) data augmentation. To ensure evaluation fidelity,
all generated samples were blindly re-graded by an IELTS expert.

## Data Schema

| Column | Description |
| :--- | :--- |
| `Question` | IELTS-style writing prompt |
| `Essay` | Generated essay text |
| `label_ovr` | Intended target band (synthetic training label) |
| `tr` | Expert-evaluated Task Response score |
| `cc` | Expert-evaluated Coherence and Cohesion score |
| `lr` | Expert-evaluated Lexical Resource score |
| `gr` | Expert-evaluated Grammatical Range and Accuracy score |
| `ovr` | Expert-evaluated overall band |

## Label Alignment

| Target Label | Expert-Evaluated Scores | Alignment Assessment |
| :--- | :--- | :--- |
| **5** | 4, 3, 4 | **Conservative Bias:** Generates below target within this audited subset. Requires relabeling or filtering. |
| **9** | 8, 8.5 | **Near-Target:** High linguistic quality, but slight task response deficits prevent perfect alignment. |

- **Band 5 Targets:** The model exhibits a conservative generation bias,
  effectively capturing lower-proficiency traits but evaluating 1-2 bands below
  the prompt according to expert grading.
- **Band 9 Targets:** The model demonstrates robust linguistic capabilities,
  frequently scoring 9s in grammar and vocabulary, but expert evaluation
  constrains overall scores to 8-8.5 due to lagging task fulfillment.

## Generative Strengths

- **High-Band Linguistic Fidelity:** Successfully synthesizes the advanced
  vocabulary and complex syntactic structures required for top-tier rubrics.
- **Authentic Error Replication:** Accurately simulates the grammatical
  constraints, simpler vocabulary, and cohesive breakdowns characteristic of
  authentic learner writing at lower bands.

## Methodological Considerations

- **Observed Conservative Scoring:** Within this audited subset, the generative
  model evaluates below the explicitly prompted target band when verified by a
  human expert.
- **Task Response Degradation:** In high-band generations, substantive argument
  development can lag behind mechanical linguistic proficiency.
- **Calibration Requirement:** Because prompted labels do not perfectly match
  human-evaluated quality, raw samples should undergo post-generation validation
  or relabeling before being used as ground-truth training data.

## Conclusion

Synthetic generation effectively replicates targeted proficiency markers for
minority band augmentation. However, inherent generative variance, confirmed via
expert re-grading in this audit, requires that samples undergo validation and
relabeling prior to integration into the AES training pipeline.
