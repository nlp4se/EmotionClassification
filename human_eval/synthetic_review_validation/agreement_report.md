# Human-as-judge agreement and correctness report

- Annotators: QM, CC, MO
- Sentences: 100
- Emotions: Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation, Neutral
- Agreement unit: each (sentence, emotion) pair is an independent binary present/absent judgment.
- Correctness: precision/recall/F1 of each annotator's binary judgments against the
  generator's intended emotion (a single positive label per sentence). Recall here is
  exactly the fraction of sentences where the intended emotion is among those marked.

---

## Agreement and correctness (all annotators)

### Inter-annotator agreement (pooled across all 9 emotions x 100 sentences)

| pair   |   kappa | interpretation   |
|:-------|--------:|:-----------------|
| QM-CC  |   0.675 | substantial      |
| QM-MO  |   0.882 | almost perfect   |
| CC-MO  |   0.696 | substantial      |

Average Cohen's kappa: **0.751** (substantial)

Fleiss' kappa (all 3 annotators): **0.749** (substantial)

### Inter-annotator agreement, by emotion

| emotion      |   avg_cohen_kappa |   fleiss_kappa |   kappa[QM-CC] |   kappa[QM-MO] |   kappa[CC-MO] |
|:-------------|------------------:|---------------:|---------------:|---------------:|---------------:|
| Joy          |             0.722 |          0.724 |          0.778 |          0.710 |          0.679 |
| Trust        |             0.733 |          0.721 |          0.665 |          0.922 |          0.612 |
| Fear         |             0.768 |          0.764 |          0.668 |          0.824 |          0.811 |
| Surprise     |             0.780 |          0.764 |          0.672 |          0.947 |          0.720 |
| Sadness      |             0.765 |          0.768 |          0.736 |          0.823 |          0.736 |
| Disgust      |             0.572 |          0.558 |          0.386 |          0.853 |          0.477 |
| Anger        |             0.871 |          0.871 |          0.853 |          0.898 |          0.863 |
| Anticipation |             0.937 |          0.935 |          0.905 |          1.000 |          0.905 |
| Neutral      |             0.574 |          0.621 |          0.370 |          0.951 |          0.400 |

### Correctness vs. ground truth (pooled across all 9 emotions x 100 sentences)

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       0.896 |    0.950 | 0.922 |
| CC          |       0.696 |    0.800 | 0.744 |
| MO          |       0.877 |    0.930 | 0.903 |
| AVERAGE     |       0.823 |    0.893 | 0.856 |

### Correctness vs. ground truth, by emotion

**Joy**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       1.000 |    0.917 | 0.957 |
| CC          |       1.000 |    0.750 | 0.857 |
| MO          |       0.875 |    0.583 | 0.700 |
| AVERAGE     |       0.958 |    0.750 | 0.838 |

**Trust**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       0.786 |    1.000 | 0.880 |
| CC          |       0.500 |    1.000 | 0.667 |
| MO          |       0.688 |    1.000 | 0.815 |
| AVERAGE     |       0.658 |    1.000 | 0.787 |

**Fear**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       0.733 |    1.000 | 0.846 |
| CC          |       0.769 |    0.909 | 0.833 |
| MO          |       1.000 |    1.000 | 1.000 |
| AVERAGE     |       0.834 |    0.970 | 0.893 |

**Surprise**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       1.000 |    0.909 | 0.952 |
| CC          |       0.556 |    0.909 | 0.690 |
| MO          |       0.909 |    0.909 | 0.909 |
| AVERAGE     |       0.822 |    0.909 | 0.850 |

**Sadness**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       0.846 |    1.000 | 0.917 |
| CC          |       1.000 |    0.727 | 0.842 |
| MO          |       0.846 |    1.000 | 0.917 |
| AVERAGE     |       0.897 |    0.909 | 0.892 |

**Disgust**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       0.900 |    0.818 | 0.857 |
| CC          |       0.438 |    0.636 | 0.519 |
| MO          |       0.769 |    0.909 | 0.833 |
| AVERAGE     |       0.702 |    0.788 | 0.736 |

**Anger**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       1.000 |    0.909 | 0.952 |
| CC          |       0.846 |    1.000 | 0.917 |
| MO          |       0.917 |    1.000 | 0.957 |
| AVERAGE     |       0.921 |    0.970 | 0.942 |

**Anticipation**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       1.000 |    1.000 | 1.000 |
| CC          |       0.846 |    1.000 | 0.917 |
| MO          |       1.000 |    1.000 | 1.000 |
| AVERAGE     |       0.949 |    1.000 | 0.972 |

**Neutral**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       0.917 |    1.000 | 0.957 |
| CC          |       1.000 |    0.273 | 0.429 |
| MO          |       1.000 |    1.000 | 1.000 |
| AVERAGE     |       0.972 |    0.758 | 0.795 |

### Correctness of the majority-vote aggregated human label (2 of 3)

Per (sentence, emotion), the aggregated human label is 1 if at least
2 of the 3 annotators marked that emotion present, else 0
(for 2 annotators this means unanimous agreement, since 1 of 2 is a tie,
not a majority). Scored against the generator's intended emotion, matching
how a 2-of-3 majority-vote ground truth is normally derived from multiple
annotators.

Pooled (all 9 emotions x 100 sentences):

| annotator     |   precision |   recall |    f1 |
|:--------------|------------:|---------:|------:|
| majority_vote |       0.941 |    0.950 | 0.945 |

By emotion:

| emotion      |   precision |   recall |    f1 |
|:-------------|------------:|---------:|------:|
| Joy          |       1.000 |    0.750 | 0.857 |
| Trust        |       0.786 |    1.000 | 0.880 |
| Fear         |       1.000 |    1.000 | 1.000 |
| Surprise     |       0.909 |    0.909 | 0.909 |
| Sadness      |       1.000 |    1.000 | 1.000 |
| Disgust      |       0.833 |    0.909 | 0.870 |
| Anger        |       1.000 |    1.000 | 1.000 |
| Anticipation |       1.000 |    1.000 | 1.000 |
| Neutral      |       1.000 |    1.000 | 1.000 |

---

## Agreement and correctness (excluding CC)

### Inter-annotator agreement (pooled across all 9 emotions x 100 sentences)

| pair   |   kappa | interpretation   |
|:-------|--------:|:-----------------|
| QM-MO  |   0.882 | almost perfect   |

Average Cohen's kappa: **0.882** (almost perfect)

Fleiss' kappa (all 2 annotators): **0.882** (almost perfect)

### Inter-annotator agreement, by emotion

| emotion      |   avg_cohen_kappa |   fleiss_kappa |   kappa[QM-MO] |
|:-------------|------------------:|---------------:|---------------:|
| Joy          |             0.710 |          0.709 |          0.710 |
| Trust        |             0.922 |          0.922 |          0.922 |
| Fear         |             0.824 |          0.823 |          0.824 |
| Surprise     |             0.947 |          0.947 |          0.947 |
| Sadness      |             0.823 |          0.823 |          0.823 |
| Disgust      |             0.853 |          0.853 |          0.853 |
| Anger        |             0.898 |          0.898 |          0.898 |
| Anticipation |             1.000 |          1.000 |          1.000 |
| Neutral      |             0.951 |          0.951 |          0.951 |

### Correctness vs. ground truth (pooled across all 9 emotions x 100 sentences)

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       0.896 |    0.950 | 0.922 |
| MO          |       0.877 |    0.930 | 0.903 |
| AVERAGE     |       0.887 |    0.940 | 0.913 |

### Correctness vs. ground truth, by emotion

**Joy**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       1.000 |    0.917 | 0.957 |
| MO          |       0.875 |    0.583 | 0.700 |
| AVERAGE     |       0.938 |    0.750 | 0.828 |

**Trust**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       0.786 |    1.000 | 0.880 |
| MO          |       0.688 |    1.000 | 0.815 |
| AVERAGE     |       0.737 |    1.000 | 0.847 |

**Fear**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       0.733 |    1.000 | 0.846 |
| MO          |       1.000 |    1.000 | 1.000 |
| AVERAGE     |       0.867 |    1.000 | 0.923 |

**Surprise**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       1.000 |    0.909 | 0.952 |
| MO          |       0.909 |    0.909 | 0.909 |
| AVERAGE     |       0.955 |    0.909 | 0.931 |

**Sadness**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       0.846 |    1.000 | 0.917 |
| MO          |       0.846 |    1.000 | 0.917 |
| AVERAGE     |       0.846 |    1.000 | 0.917 |

**Disgust**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       0.900 |    0.818 | 0.857 |
| MO          |       0.769 |    0.909 | 0.833 |
| AVERAGE     |       0.835 |    0.864 | 0.845 |

**Anger**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       1.000 |    0.909 | 0.952 |
| MO          |       0.917 |    1.000 | 0.957 |
| AVERAGE     |       0.958 |    0.955 | 0.954 |

**Anticipation**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       1.000 |    1.000 | 1.000 |
| MO          |       1.000 |    1.000 | 1.000 |
| AVERAGE     |       1.000 |    1.000 | 1.000 |

**Neutral**

| annotator   |   precision |   recall |    f1 |
|:------------|------------:|---------:|------:|
| QM          |       0.917 |    1.000 | 0.957 |
| MO          |       1.000 |    1.000 | 1.000 |
| AVERAGE     |       0.958 |    1.000 | 0.978 |

### Correctness of the majority-vote aggregated human label (2 of 2)

Per (sentence, emotion), the aggregated human label is 1 if at least
2 of the 2 annotators marked that emotion present, else 0
(for 2 annotators this means unanimous agreement, since 1 of 2 is a tie,
not a majority). Scored against the generator's intended emotion, matching
how a 2-of-3 majority-vote ground truth is normally derived from multiple
annotators.

Pooled (all 9 emotions x 100 sentences):

| annotator     |   precision |   recall |    f1 |
|:--------------|------------:|---------:|------:|
| majority_vote |       0.958 |    0.910 | 0.933 |

By emotion:

| emotion      |   precision |   recall |    f1 |
|:-------------|------------:|---------:|------:|
| Joy          |       1.000 |    0.583 | 0.737 |
| Trust        |       0.786 |    1.000 | 0.880 |
| Fear         |       1.000 |    1.000 | 1.000 |
| Surprise     |       1.000 |    0.909 | 0.952 |
| Sadness      |       1.000 |    1.000 | 1.000 |
| Disgust      |       0.900 |    0.818 | 0.857 |
| Anger        |       1.000 |    0.909 | 0.952 |
| Anticipation |       1.000 |    1.000 | 1.000 |
| Neutral      |       1.000 |    1.000 | 1.000 |

