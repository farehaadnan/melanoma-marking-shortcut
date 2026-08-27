# Do Melanoma Classifiers Learn to Read Ink Marks Instead of Lesions?

A reproduction and mitigation study built on:

> Winkler JK, Fink C, Toberer F, Enk A, Deinlein T, Hofmann-Wellenhof R, Thomas L, Lallas A, Blum A, Stolz W, Haenssle HA. **Association Between Surgical Skin Markings in Dermoscopic Images and Diagnostic Performance of a Deep Learning Convolutional Neural Network for Melanoma Recognition.** *JAMA Dermatology.* 2019;155(10):1135–1141. doi:[10.1001/jamadermatol.2019.1735](https://doi.org/10.1001/jamadermatol.2019.1735)

## The finding we're building on

Winkler et al. tested a clinically-used melanoma-detection CNN on 130 lesions, in two versions: as photographed, and with a doctor's surgical ink marking digitally added. On the same lesions:

| | Sensitivity | Specificity |
|---|---|---|
| Unmarked | 95.7% | 84.1% |
| Marked | 100% | **45.8%** |

Adding a mark — with the underlying lesion completely unchanged — nearly halved the model's ability to correctly call something benign. Their paper stops at documenting this and recommending marks be avoided in clinical photos; it doesn't attempt to fix the underlying model.

## What this project does

1. **Reproduces the mechanism**, not just the correlation. Trains a ResNet18 on ISIC-2017 under two conditions:
   - *Natural* training data (no injected mark/label correlation) — used as a control.
   - *Deliberately biased* training data, where synthetic marks are added to 75% of melanoma images and 5% of nevus images during training, mimicking the real-world correlation between clinical suspicion and marking.
2. **Confirms it's causal, not just visual clutter**, by testing both models on the same held-out images, clean vs. synthetically marked.
3. **Fixes it** with counterfactual augmentation: retraining with marks added to *both* classes at random with equal probability, removing any information the mark could carry about the label.
4. **Re-measures** the fixed model under the same clean/marked test to check the fix actually closes the gap, without sacrificing normal accuracy.

Synthetic marking was chosen over sourcing real marked photos because the original paper validated that electronically-added marks produce results statistically indistinguishable from real ink markings (P = .78 between the two conditions in their study).

## Results

| Model | Clean Sens. | Clean Spec. | Marked Sens. | Marked Spec. | Clean→Marked spec. drop |
|---|---|---|---|---|---|
| Natural baseline | 0.786 | 0.690 | 0.744 | 0.656 | −0.034 |
| **Biased** (injected correlation) | 0.701 | 0.784 | 0.932 | 0.267 | **−0.517** |
| **Fixed** (counterfactual augmentation) | 0.701 | 0.814 | 0.667 | 0.804 | **−0.010** |

The natural baseline — never exposed to a mark/label correlation during training — is only mildly affected by marks at test time. Once that correlation is deliberately injected during training, the model exhibits the same qualitative failure Winkler et al. reported: specificity collapses (marked sensitivity actually *rises*, because the model is simply calling "melanoma" whenever a mark is present, which trivially catches more true melanomas while destroying its ability to correctly clear benign cases). Counterfactual augmentation during training closes the gap almost entirely.

### Qualitative example

One nevus, three models, same synthetic marking:

| Model | Prediction | P(melanoma) |
|---|---|---|
| Baseline | Nevus | 3.6% |
| Biased | **Melanoma** | 73.6% |
| Fixed | Nevus | 1.4% |

*(See `gradcam_comparison.png` for the Grad-CAM panel — insert the 4-image comparison here.)*

## Repository contents

- `melanoma_marking_shortcut.ipynb` — the full notebook, runnable top-to-bottom on a free-tier Colab GPU.
- `gradcam_comparison.png` — Grad-CAM comparisons and result plots referenced above.

## How to run

1. Open the notebook in Google Colab.
2. Run cells top to bottom. It downloads ISIC-2017 automatically (images + classification labels).
3. Training the three models (baseline, biased, fixed) takes roughly 15–30 minutes total on a T4.

## Limitations

- **Different model, different data.** This uses a from-scratch fine-tuned ResNet18 on the public ISIC-2017 dataset, not the original paper's market-approved commercial CNN or their 130-lesion clinical set. The goal is testing whether the *mechanism* reproduces on independent data/model, not replicating their exact numbers.
- **Synthetic markings**, not real clinical ink — justified by the original paper's own validation that the two are statistically equivalent, but still a simplification worth stating plainly.
- **Modest test set.** ISIC-2017's melanoma-vs-nevus test split is small; report raw case counts alongside percentages when citing these numbers, since small samples move in large discrete steps.
- The biased-training condition uses a deliberately strong injected correlation (75% vs. 5%) to make the causal mechanism unambiguous — larger than what likely occurs naturally in clinical data, so the resulting specificity collapse is not directly comparable in magnitude to the original paper's.

## Citation

If referencing this work, please also cite the original paper this builds on:

```bibtex
@article{winkler2019association,
  title={Association Between Surgical Skin Markings in Dermoscopic Images and Diagnostic Performance of a Deep Learning Convolutional Neural Network for Melanoma Recognition},
  author={Winkler, Julia K and Fink, Christine and Toberer, Ferdinand and Enk, Alexander and Deinlein, Teresa and Hofmann-Wellenhof, Rainer and Thomas, Luc and Lallas, Aimilios and Blum, Andreas and Stolz, Wilhelm and Haenssle, Holger A},
  journal={JAMA Dermatology},
  volume={155},
  number={10},
  pages={1135--1141},
  year={2019},
  doi={10.1001/jamadermatol.2019.1735}
}
```
