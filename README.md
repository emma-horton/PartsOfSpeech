
![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=flat)  ![NLTK Badge](https://img.shields.io/badge/NLTK-3776AB?logo=nltk&logoColor=white&style=flat)  ![CoNLLU Badge](https://img.shields.io/badge/CoNLLU-3776AB?logo=conllu&logoColor=white&style=flat)  ![Dynamic Programming Badge](https://img.shields.io/badge/Dynamic_Programming-3776AB?logo=dynamicprogramming&logoColor=white&style=flat)  
# **Part-of-Speech Tagging with Dynamic Algorithms**  

---

## **Overview**  
This project explores the implementation and comparison of three Part-of-Speech (POS) tagging algorithms—**Eager**, **Viterbi**, and **Individually Most Probable Tags**—across English, Swedish, and Korean. These algorithms were designed to navigate the complexities of morphology and syntax in different languages, revealing intriguing patterns in linguistic structure and algorithm performance.

## **Project Goals**  
1. Implement three distinct POS tagging algorithms of varying complexity: **Eager**, **Viterbi**, and **Individually Most Probable Tags**.  
2. Train and evaluate these algorithms using multilingual corpora from the [Universal Dependencies Treebank](https://universaldependencies.org).  
3. Uncover linguistic insights by analyzing algorithm performance across English, Swedish, and Korean.  


## **Key Findings**  
### Algorithm Performance at a Glance  
| **Language** | **Eager Accuracy (%)** | **Viterbi Accuracy (%)** | **Individually Most Probable Tags Accuracy (%)** |
|--------------|--------------------------|---------------------------|-----------------------------------------------| 
| **English**  | 88.6                    | 91.3                      | 88.6                                          |
| **Swedish**  | 85.7                    | 90.2                      | 85.7                                          |
| **Korean**   | 80.8                    | 79.2                      | 80.8                                          |


## **How to Use This Project**  
### **1. Install Dependencies**  
```bash
pip install conllu
pip install nltk
```

### **2. Run the Script**  
```bash
python3 pos_tagging.py
```

## **Technologies Used**  
- **Python**: Primary programming language for implementation.  
- **CoNLL-U**: For parsing and preparing corpora.  
- **NLTK**: To calculate emission and transition probabilities.  


## **Acknowledgements**  
Grateful for the [Universal Dependencies Treebank](https://universaldependencies.org) for providing high-quality multilingual data, enabling this exploration into the intricacies of POS tagging.  


## **Want to fin out more?**  

For more insights, read the associated blog post:  
[**Decoding Language: A Deep Dive into Part-of-Speech Tagging**](https://emmahorton.me/projects/PartsOfSpeechBlog/PartsOfSpeechBlog.html)  

