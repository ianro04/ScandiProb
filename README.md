# ScandiProb
Work in Progress. Code for fine-tuning [ScandiBERT](https://huggingface.co/vesteinn/ScandiBERT) (by Snæbjarnarson, et al.) to be a language ID text classifier, reporting independent probabilities that a given text is: Norwegian, Swedish, and Danish. To be hosted on HuggingFace. Corpus: [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)

Completed in iterations from Part D to Part A, as follows.

<h2> Part D: Zero-Shot/Few-Shot Norwegian Classification </h2>

  - [X] Load ScandiBERT into Kaggle Notebooks. Attach a classification head to it, using a sigmoid function
to output the probability between 0 and 1 that a given text is Norwegian or not.
  - [X] Implement a bare bones negative class with just regex and an equation of (1 - p<sub>n</sub>).
  - [X] Run tests to see how successful the model is with just pretraining and just one language.
<h2> Part C: Training for Norwegian, Swedish, and the Negative Class </h2>
  
  - [X] Properly train the model on limited labeled corpora of labeled Norwegian, Swedish, and a limited non-Scandinavian corpus.
  - [X] Adjust the classification head to now return three independent probabilities for whether a text is in Norwegian, Swedish, or non-Scandinavian.
  - [X] Run new tests for all three classes.

<h2> Part B: Training for Danish </h2>

  - [X] Properly train the model on extensive labeled corpora of Norwegian, Swedish, and Danish.
  - [X] Adjust the classification head to now return an independent probability for Danish in addition to existing classes. Note that Danish is likely the hardest class to implement properly, due to significant overlap with Norwegian Bokmål.

<h2> Part A: Options for Expansion </h2>
Complete two of the following:

  - [ ] Implement a proper UI for the final program. Have the user input/upload their text and have it output a bar graph for the language probabilities.
  - [ ] Retrain the Norwegian class to have multiple subclasses for Bokmål, Nynorsk, and "Miscellaneous Norwegian Dialect". ScandiBERT
is only pretrained on Bokmål and Nynorsk and does not label them as part of Norwegian at-large. NorDial is a corpus of written Norwegian dialects and will help to make the Norwegian class as accurate as possible.
  - [ ] After the model outputs the probabilities, write a function that goes character-bycharacter and adds or removes language score
based on linguistic heuristics, such as Danish having -d at the end of words that gets dropped off in Norwegian Bokmål. Does this
meaningfully improve accuracy? This may be *necessary* to some extent to distinguish between Danish and Bokmål.
  - [ ] Implement classes and probabilities for the insular Scandinavian languages of Icelandic and Faroese, as ScandiBERT is pretrained on them.
