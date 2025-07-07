# Lexical Substitution System

This project implements several approaches for the Lexical Substitution task—a key problem in lexical semantics and word sense disambiguation. Given a target word in context, the system suggests suitable substitutes using both classical and neural NLP techniques.

## Overview

The system proposes context-appropriate substitutes for a given word in a sentence, evaluated on standard lexical substitution datasets.

## Features

- **WordNet-based substitution:** Leverages lexical relations and sense filtering.
- **Word2Vec similarity:** Ranks candidate words by cosine similarity in embedding space.
- **BERT masked language modeling:** Generates context-aware substitutes using a transformer model.
- **Evaluation scripts:** Compare predictions to gold-standard answers.

## Directory Structure

```
hw4_files/
├── lexsub_main.py                   # Main entry point for substitution
├── lexsub_xml.py                    # XML parsing utilities
├── lexsub_trial.xml                 # Example input file
├── gold.trial                       # Gold-standard evaluation file
├── GoogleNews-vectors-negative300.bin.gz  # Pre-trained Word2Vec embeddings
├── part2.predict                    # Example output (Word2Vec)
├── part3.predict                    # Example output (BERT)
└── ...                              # Additional scripts and data
```

## Dependencies

Install all required packages with:

```bash
pip install nltk numpy gensim transformers torch
```

Download required NLTK data:

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"
```

**Notes:**  
- For BERT-based methods, a GPU is recommended but not required.
- The Word2Vec model file (`GoogleNews-vectors-negative300.bin.gz`) is large (~1.5GB).

## Downloading Pre-trained Word2Vec Embeddings

This project requires the pre-trained Google News Word2Vec embeddings (`GoogleNews-vectors-negative300.bin.gz`).  
Due to its large size, this file is **not included in the repository**.

You can download it from the official Gensim repository:

- [GoogleNews-vectors-negative300.bin.gz (1.5GB)](https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz)

After downloading, place the file in the same directory as `lexsub_main.py`.

## Running the System

Navigate to the `hw4_files/` directory and run:

```bash
python lexsub_main.py --input <input_file> --output <output_file> --method <method>
```

- `<input_file>`: Path to the input XML or text file (e.g., `lexsub_trial.xml`)
- `<output_file>`: Where to write predictions (e.g., `my_output.txt`)
- `<method>`: One of `wordnet`, `w2v`, or `bert`

**Example:**

```bash
python lexsub_main.py --input lexsub_trial.xml --output my_output.txt --method bert
```

## Technical Challenges

- **Large Model Handling:** Loading the Google News Word2Vec model requires significant RAM and disk space.
- **Contextual Disambiguation:** BERT-based substitution must accurately mask and predict the correct word in context, which is non-trivial for ambiguous sentences.
- **Integration of Resources:** Combining lexical databases (WordNet), static embeddings (Word2Vec), and contextual models (BERT) required careful data handling and preprocessing.
- **Performance:** Achieving competitive precision/recall is challenging due to the open-ended nature of lexical substitution and the limitations of each approach.

## Output & Evaluation

- The system outputs one substitute per target instance, formatted for evaluation.
- Use the provided `gold.trial` file to compare your predictions.

## References

- [SemEval 2007 Task 10: Lexical Substitution](https://www.cs.york.ac.uk/semeval-2007/tasks/task10/)
- [NLTK Documentation](https://www.nltk.org/)
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
- [HuggingFace Transformers](https://huggingface.co/transformers/)

---

*This repository demonstrates several approaches to context-aware lexical substitution using modern NLP tools and resources.* 