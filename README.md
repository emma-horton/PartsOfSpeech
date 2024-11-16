
![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=flat)  ![NLTK Badge](https://img.shields.io/badge/NLTK-3776AB?logo=nltk&logoColor=white&style=flat)  ![CoNLLU Badge](https://img.shields.io/badge/CoNLLU-3776AB?logo=conllu&logoColor=white&style=flat)  ![Dynamic Programming Badge](https://img.shields.io/badge/Dynamic_Programming-3776AB?logo=dynamicprogramming&logoColor=white&style=flat)  
# **Part-of-Speech Tagging with Dynamic Algorithms**  
Harnessing the power of Python and computational linguistics to decode language!

---

## **Overview**  
This project explores the implementation and comparison of three Part-of-Speech (POS) tagging algorithms—**Eager**, **Viterbi**, and **Individually Most Probable Tags**—across English, Swedish, and Korean. These algorithms were designed to navigate the complexities of morphology and syntax in different languages, revealing intriguing patterns in linguistic structure and algorithm performance.

---

## **Key Findings**  
### Algorithm Performance at a Glance  
| **Language** | **Eager Accuracy (%)** | **Viterbi Accuracy (%)** | **Individually Most Probable Tags Accuracy (%)** |
|--------------|--------------------------|---------------------------|-----------------------------------------------| 
| **English**  | 88.6                    | 91.3                      | 88.6                                          |
| **Swedish**  | 85.7                    | 90.2                      | 85.7                                          |
| **Korean**   | 80.8                    | 79.2                      | 80.8                                          |

- **Viterbi excels** in English and Swedish due to their rigid syntactic structure and predictable patterns.  
- **Eager performs best in Korean**, highlighting its strength in handling agglutinative languages with rich morphological cues.  
- The **Individually Most Probable Tags algorithm**, though innovative, fell short of expectations and warrants further refinement.  

### Language Complexity and Morphology  
- **English**: Simpler morphology and predictable word order lead to the highest accuracy.  
- **Swedish**: Moderate inflectional complexity benefits from Viterbi’s context-aware approach.  
- **Korean**: Agglutinative structure with flexible word order poses challenges for sentence-wide models like Viterbi.  

---

## **Project Goals**  
1. Implement three distinct POS tagging algorithms of varying complexity: **Eager**, **Viterbi**, and **Individually Most Probable Tags**.  
2. Train and evaluate these algorithms using multilingual corpora from the [Universal Dependencies Treebank](https://universaldependencies.org).  
3. Uncover linguistic insights by analyzing algorithm performance across English, Swedish, and Korean.  

---

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

---

## **Technologies Used**  
- **Python**: Primary programming language for implementation.  
- **CoNLL-U**: For parsing and preparing corpora.  
- **NLTK**: To calculate emission and transition probabilities.  

---

## **Reflections and Challenges**  
- Translating theoretical algorithms, especially **Viterbi**, into Python required meticulous iteration and debugging.  
- The **Individually Most Probable Tags** algorithm involved advanced probabilistic modeling, demanding a deep dive into mathematical formulations.  
- Korean’s rich morphology highlighted the importance of adapting algorithms to language-specific features.  

---

## **Looking Ahead**  
This project underscores the potential of dynamic algorithms for tackling complex linguistic tasks. Future work will focus on refining the **Individually Most Probable Tags** algorithm and exploring custom adaptations for agglutinative languages like Korean.  

---

## **Acknowledgements**  
Grateful for the [Universal Dependencies Treebank](https://universaldependencies.org) for providing high-quality multilingual data, enabling this exploration into the intricacies of POS tagging.  

---

## **Want to fin out more?**  

For more insights, read the associated blog post:  
[**Decoding Language: A Deep Dive into Part-of-Speech Tagging**](https://emmahorton.me/projects/PartsOfSpeechBlog/PartsOfSpeechBlog.html)  

---

Crafted with curiosity and Python 🐍  
By Emma Horton  
