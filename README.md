# ScandiProb: Hybrid Language ID Classifier
Code for fine-tuning [ScandiBERT](https://huggingface.co/vesteinn/ScandiBERT) (by Snæbjarnarson, et al.) to be a hybrid language ID text classifier, reporting independent probabilities that a given text is: Norwegian, Swedish, Danish, and Non-Scandinavian / None of the Above. Code was primarily written in Kaggle Notebooks (scandiprob.ipynb), then committed to GitHub with each iteration, and finally hosted on HuggingFace. This was done as an undergraduate final project for a Spring 2026 NLP course at the University of Alaska Fairbanks. You can use the resulting web app [here](https://huggingface.co/spaces/ianro04/ScandiProb).

The program turns ScandiBERT into a multilabel classifier, trained on limited amounts of translation corpus OPUS-100 (6000 samples per label) and a tiny handwritten multilabel dataset, and combined with regex-enforced heuristics. Achieves ~93% macro-F1 score on OPUS-100 test set and ~84% macro-F1 score against the comprehensive SLIDE eval set, with a fraction of the training data used in the 2025 SLIDE paper.

([Kaggle Notebooks](https://www.kaggle.com/code/cloudeerie/scandiprob) | [Hugging Face Model Page](https://huggingface.co/ianro04/ScandiProb))

<h2>Project Progression / Rubric</h2>
Completed in iterations from Part D to Part A, as follows.

<h3> Part D: Zero-Shot/Few-Shot Norwegian Classification </h3>

  - [X] Load ScandiBERT into Kaggle Notebooks. Attach a classification head to it, using a sigmoid function
to output the probability between 0 and 1 that a given text is Norwegian or not.
  - [X] Implement a bare bones negative class with just regex and an equation of (1 - p<sub>n</sub>).
  - [X] Run tests to see how successful the model is with just pretraining and just one language.
<h3> Part C: Training for Norwegian, Swedish, and the Negative Class </h3>
  
  - [X] Properly train the model on limited labeled corpora of labeled Norwegian, Swedish, and a limited non-Scandinavian corpus.
  - [X] Adjust the classification head to now return three independent probabilities for whether a text is in Norwegian, Swedish, or non-Scandinavian.
  - [X] Run new tests for all three classes.

<h3> Part B: Training for Danish </h3>

  - [X] Properly train the model on extensive labeled corpora of Norwegian, Swedish, and Danish.
  - [X] Adjust the classification head to now return an independent probability for Danish in addition to existing classes. Note that Danish is likely the hardest class to implement properly, due to significant overlap with Norwegian Bokmål.

<h3> Part A: Options for Expansion </h3>
Complete two of the following for completion:

*Italicized weren't pursued or completed.*

  - [X] Implement a proper UI for the final program. Have the user input/upload their text and have it output a bar graph for the language probabilities.
  - [ ] *Retrain the Norwegian class to have multiple subclasses for Bokmål, Nynorsk, and "Miscellaneous Norwegian Dialect". ScandiBERT
is only pretrained on Bokmål and Nynorsk and does not label them as part of Norwegian at-large. NorDial is a corpus of written Norwegian dialects and will help to make the Norwegian class as accurate as possible.*
  - [X] After the model outputs the probabilities, write a function that goes character-by-character and adds or removes language score
based on linguistic heuristics, such as Danish having -d at the end of words that gets dropped off in Norwegian Bokmål. Does this
meaningfully improve accuracy? This may be *necessary* to some extent to distinguish between Danish and Bokmål.
  - [ ] *Implement classes and probabilities for the insular Scandinavian languages of Icelandic and Faroese, as ScandiBERT is pretrained on them.*

<h2>References</h2>

*Formal references to be found in Hugging Face raw model README.md*

Base Model: [ScandiBERT](https://huggingface.co/vesteinn/ScandiBERT) by Snæbjarnarson, et al., 2023.

Training, Validation, and Test Data: [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) by Zhang, et al.

Final Validation Dataset for Comparison: [SLIDE](https://huggingface.co/datasets/ltg/slide) by Fedorova, et al., 2025.

<h5>General References on Project Methodology</h5>

[Multi-label Scandinavian Language Identification (SLIDE)](https://aclanthology.org/2025.resourceful-1.33.pdf) by Fedorova, et al., 2025.

[Transfer to a Low-Resource Language via Close Relatives: The Case Study on Faroese](https://aclanthology.org/2023.nodalida-1.74.pdf) by by Snæbjarnarson, et al., 2023.

[Discriminating between Similar Nordic Languages](https://aclanthology.org/2021.vardial-1.8.pdf) by Haas & Derczynski, 2021.
