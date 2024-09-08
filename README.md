# Parts of Speech Tagging with Dynamic Algorithms

![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=flat)
![NLTK Badge](https://img.shields.io/badge/NLTK-3776AB?logo=nltk&logoColor=white&style=flat)
![CoNLLU Badge](https://img.shields.io/badge/CoNLLU-3776AB?logo=conllu&logoColor=white&style=flat)
![Dynamic Programming Badge](https://img.shields.io/badge/Dynamic_Programming-3776AB?logo=dynamicprogramming&logoColor=white&style=flat)

## Overview
This project was developed for the “Language and Computation” module at the University of St Andrews as an individual assignment. The program, written in Python, utilizes data collected from UD Treebanks. The data was processed using the CoNLL-U package, and the model was implemented with the help of NLTK.

## Purpose 
To implement and compare the Viterbi algorithm with the Eager and Individually Most Probable Tags algorithms to optimize part of speech tagging accuracy.

## Features 
* Implementation of three algorithms with varying complexity: Eager, Viterbi, and Individually Most Probable Tags.
* Training on corpora in three distinct languages: English, Swedish, and Korean.
* Evaluation of part-of-speech tagging accuracy for each language using unseen test sets.

## Usage
#### 1. Install dependancies 
```bash
pip install conullu
npm install nltk
```
#### 2. Run script
```bash
python3 p1.py
```

## Technologies Used 
* **CoNLL-U**: Utilized for parsing and organizing corpora from the UD Treebank into training and testing sets.
* **NLTK**: Employed to compute emission and transition probabilities essential for training the models.

## Acknowledgements
* [UD Treebank](https://universaldependencies.org): Credited for supplying the data used to train and test the models across various languages.

