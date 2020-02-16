# Twitter Viterbi

Part-of-Speech tagger for tweets using the Viterbi Algorithm.

Includes a vector-optimized implementation that pre-computes all transition probabilities and iterates through a sequence, calculating the optimal path using NumPy matrix broadcasting in log space.

## Getting Started

Create a new environment for the project
```
conda create --name twitter-viterbi python=3.7
```

and activate it
```
conda activate twitter-viterbi
```

Install AllenNLP and NumPy
```
pip install -r requirements.txt
```

Test your installation by running the unit tests:
```
pytest -v -W ignore
```

# Acknowledgements

Special thanks to [Andrew Li](https://github.com/lia4) for helping me understand the implementation of Viterbi with matrix broadcasting. 
