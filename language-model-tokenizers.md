<link href="styles.css" rel="stylesheet"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>

# LLM Tokenizations: Understanding Text Tokenizations for Transformer-based Language Models

<p style="opacity: 0.5;">25 December, 2023</p>
<hr>

- [Introduction](#introduction)
- [Word Tokenization](#word-tokenization)
- [Character-level Tokenization](#character-level-tokenization)
- [Subword Tokenization](#word-level-tokenization)
    - [Byte-Pair Tokenization](#byte-pair-tokenization)
    - [WordPiece Tokenization](#wordpiece-tokenization)
    - [Unigram Tokenization](#unigram-tokenization)
    - [SentencePiece Tokenization](#sentencepiece-tokenization)
- [Summary](#summary)

## Introduction

As it is established practice that Computers represent the data digitally in the binary format. Similarly, Large Language Models (LLMs) do not directly process the text provided by the users in the shape of a prompt. The user prompts are passed through a multi-stage pre-processing pipeline prior to being fed to a LLM. Furthermore, you may have came across the "tokens" while navigating the pricing pages of the popular commercial LLM service providers.


In this post, we will expolore various tokenization mechanisms. Including the ones used by the main stream LLMs including LLAMA, GPT, BERT etc. Alongside the simulated tokenization dry runs, we will also go over the implementation of the such tokenization algorithms alongside. We can begin by looking the tokenizers tree, which is composed of most prominent tokenization algorithms.

![alt text](/media/language-model-tokenizers/tokenizers-tree.png "Tokenizers Tree")

<blockquote class="blockstyle">
 <span class="triangle">💡</span> It is pertinent to note that tokenization process directly influences the items in a vocabulary (formally called <em>word types</em>)of a language model alongside the size of the vocabulary. Additionally, each word type from the vocabulary corresponds to its counterpart word embedding latent vectors in the embedding matrix. Concretely, the size of vocabulary is directly proportional to number of embedding vectors in the embedding matrix of a language model. Thereby, you may also prune the vocabulary to discard the infrequent <em>word types</em> post-tokenization to reduce the size of the embedding matrix (which contains trainable weights). 
</blockquote>

## Word Tokenization
## Character-level Tokenization
## Subword Tokenization
### Byte-Pair Tokenization
### WordPiece Toekenization
### Unigram Tokenization
### SentencePiece Tokenization
