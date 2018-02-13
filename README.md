# Viterbi using Trigram Hidden Markov Models on Twitter Data

Part-of-Speech tagger for tweets using the Viterbi Algorithm.

Includes a vector-optimized implementation that pre-computes all transition probabilities and iterates through a sequence, calculating the optimal path using NumPy matrix broadcasting in log space.

Special thanks to Andrew Li (https://github.com/lia4) for explanations of vectorized Viterbi decoding. 

# Getting Started

### Prequisites

* Python 3 (preferrably 3.6.3)
* NumPy

### Code Explained

* symbols.py — Contains the defined UNKs as well as a function that UNKs a word given an effective vocabulary.  Note that only the trigram Viterbi utilized all UNKs; bigram Viterbi utilized only two. Reasons for this as well as other design choices can be found in the writeup.
* util.py — Contains a scoring function shared throughout the Viterbi code
* HMM.py — A Bigram Hidden Markov Model smoothed with Linear Interpolation
* TrigramHMM.py — A Trigram Hidden Markov Model smoothed with Linear Interpolation
* trigram_viterbi.py — Base implementation of Viterbi using trigram HMMs.
* trigram_viterbi_vectorized.py — A vector-optimized version of trigram_viterbi.py using matrix operations instead of loops. Achieves the same accuracy as the base implementation.

## Training
Training examples should come from a json file where each line is of the form

``[["word", "part-of-speech-tag"], ..., ["word", "part-of-speech-tag"]``
This implementation is intended for part-of-speech tagged Twitter data available here: https://code.google.com/archive/p/ark-tweet-nlp/downloads

## Running the Trigram Viterbi Code

To run the base implementation of the trigram Viterbi algorithm to label all sentences in a corpus and calculate accuracy, run

​	`python trigram_viterbi.py <path to twt.train.json> <path to twt.(dev|test).json>`

**Note** that this version takes approximately 15-30 minutes to complete on the development set.

To run the optimized implementation of the trigram Viterbi algorithm, run

​	`python trigram_viterbi_vectorized.py <path to twt.train.json> <path to twt.(dev|test).json>`

**This** version should take approximately 20 seconds to complete.


#### **All** versions of Viterbi above will print the time taken to complete as well as the per-word accuracy over the validation set given.
