# Viterbi for Bigrams and Trigrams on Twitter Data

Tam Dang
CSE 447 - Hidden Markov Models
2.4.2018


Part-of-Speech tagger for tweets using the Viterbi Algorithm.



# Getting Started

### Prequisites

* Python 3 (preferrably 3.6.3)
* NumPy

Please **do not alter the file structure**, the hidden markov models and Viterbi code rely on the symbols and utilities code to be in the same directory.

### Code Explained

* symbols.py — Contains the defined UNKs as well as a function that UNKs a word given an effective vocabulary.  Note that only the trigram Viterbi utilized all UNKs; bigram Viterbi utilized only two. Reasons for this as well as other design choices can be found in the writeup.
* util.py — Contains a scoring function shared throughout the Viterbi code
* HMM.py — A Bigram Hidden Markov Model smoothed with Linear Interpolation
* TrigramHMM.py — A Trigram Hidden Markov Model smoothed with Linear Interpolation
* bigram_viterbi.py — Implementation of Viterbi using bigram HMMs.
* trigram_viterbi.py — Base implementation of Viterbi using trigram HMMs.
* trigram_viterbi_vectorized.py — A vector-optimized version of trigram_viterbi.py using matrix operations instead of loops. Achieves the same accuracy as the base implementation.




## Running the Bigram Viterbi Code

To run the bigram Viterbi algorithm, run

​	`python bigram_viterbi.py <path to twt.train.json> <path to twt.(dev|test).json>`

For error analysis, the top 20 mislabled instances will be printed before overall accuracy is reported.

## Running the Trigram Viterbi Code

To run the base implementation of the trigram Viterbi algorithm, run

​	`python trigram_viterbi.py <path to twt.train.json> <path to twt.(dev|test).json>`

**Note** that this version takes approximately 15-30 minutes to complete on the development set.

To run the optimized implementation of the trigram Viterbi algorithm, run

​	`python trigram_viterbi_vectorized.py <path to twt.train.json> <path to twt.(dev|test).json>`

**This** version should take approximately 20 seconds to complete.



**Don't worry about the runtime warning about division by zero:** Python and NumPy will replace division by zero values with negative infinity (float('-inf')) appropriately.



#### **All** versions of Viterbi above will print the time taken to complete as well as the per-word accuracy over the validation set given.
