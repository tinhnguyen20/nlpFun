# Fun with NLP - Indian Law Judgement classification
My first crack at NLP. I compared two basic text classification models: 
 - Naive Bayes
  - Achieved ~50% with no preprocessing (1-gram, bag of words)
  - Possibly better with mutual information
 - SVM
  - Worse due to small number of samples (n << d)
  
## Feature Extraction
After some light research, I found that mutual information may help define more important words, reduce noise, and increase accuracy.

## What I'm looking to do
 - Achieve better accuracy (as always)
 - explore Chi2 feature selection

### Relevent Links
 - http://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html
 - http://nlp.stanford.edu/IR-book/html/htmledition/feature-selectionchi2-feature-selection-1.html
