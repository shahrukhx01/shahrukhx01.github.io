<link href="styles.css" rel="stylesheet"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>

# LLM Tokenizations: Understanding Text Tokenizations for Transformer-based Language Models

<p style="opacity: 0.5;">25 December, 2023</p>
<hr>

- [Introduction](#introduction)
- [Tokenization Pipelines in Practice](#tokenization-pipelines-in-practice)
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

<blockquote class="blockstyle" style="color:#00CC8F;">
 <span class="triangle">💡</span> It is pertinent to note that tokenization process directly influences the items in a vocabulary (formally called <em>word types</em>)of a language model alongside the size of the vocabulary. Additionally, each word type from the vocabulary corresponds to its counterpart word embedding latent vectors in the embedding matrix. Concretely, the size of vocabulary is directly proportional to number of embedding vectors in the embedding matrix of a language model. Thereby, you may also prune the vocabulary to discard the infrequent <em>word types</em> post-tokenization to reduce the size of the embedding matrix (which contains trainable weights). 
</blockquote>

# Tokenization Pipelines in Practice
Before we delve into aforementioned tokenization mechanism. It is important to highlight that tokenizers in practice are not used stand-alone on raw input. Rather specialized pre-processing and post-processing steps are performed to standardize tokenization results. This becomes even more critical when processing multi-lingual corpora.

![alt text](/media/language-model-tokenizers/tokenization_pipeline.svg "Tokenization Pipeline")
<em>source: https://huggingface.co/learn/nlp-course/chapter6/4</em>

## Normalization

The normalization process includes basic tidying tasks like eliminating unnecessary spaces, converting to lowercase, and/or eliminating accents. If you're acquainted with Unicode normalization (like NFC or NFKC), the tokenizer might also implement this step. Below is a simple example of a normalizer from Huggingface tokenizers.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_normalizer = tokenizer.backend_tokenizer.normalizer
print(bert_normalizer.normalize_str("Héllò hôw are ü?"))
```
<em>Output:</em>
```bash
'hello how are u?'
```

## Pre-tokenization

Training a tokenizer solely on raw text is not feasible. To address this, we use pre-tokenization to initially divide the text into smaller entities, such as words. As discussed [later](#word-tokenization), a word-based tokenizer can simply split a raw text into words on whitespace and punctuation. The resulting words become the boundaries of the subtokens the tokenizer can learn during its training.

### Pre-tokenization: BERT vs GPT vs T5
Below is a simple comparison taken from [Huggingface NLP course](https://huggingface.co/learn/nlp-course/chapter6/4?fw=pt#pre-tokenization).

```python
from transformers import AutoTokenizer

# BERT pre_tokenize
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")

# Output:
[('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]


# GPT pre_tokenize
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")

# Output:
[('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), ('Ġ', (14, 15)), ('Ġyou', (15, 19)),
 ('?', (19, 20))]

 # T5 pre_tokenize
 tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")

# Output:
[('▁Hello,', (0, 6)), ('▁how', (7, 10)), ('▁are', (11, 14)), ('▁you?', (16, 20))]
```
#### Key Highlights:
<table>
  <tr>
    <th>Tokenizer</th>
    <th>Splitting Mechanism</th>
    <th>Whitespace inclusion</th>
  </tr>
  <tr>
    <td>BERT</td>
    <td>Whitespace and punctuation</td>
    <td>Ignores</td>
  </tr>
  <tr>
    <td>GPT</td>
    <td>Whitespace and punctuation</td>
    <td>Replaces with Ġ</td>
  </tr>
  <tr>
    <td>T5</td>
    <td>Whitespace only</td>
    <td>Replaces with _ and also prepends _ at the beginning of the sequence</td>
  </tr>
</table>


## Word Tokenization
The most trivial form of tokenization is based on the splitting the text by space. For instance, you have the text `Don't you love 🤗 Transformers? We sure do.`. The word-level space-based tokenizer will split it into tokens `["Don't", "you", "love", "🤗", "Transformers?", "We", "sure", "do."]`. If you notice closely, the space-based tokenizer does not split compound words such as `Don't`, however, `Don't` corresponds to `do not`. Hence, in practice NLP libraries like [spacy](https://spacy.io/) and [Moses](https://www2.statmt.org/moses/?n=Development.GetStarted) implement word-level tokenizers as a combination of rule and space-based approaches. Thereby, these rule-based tokenizers will produce the following tokens for the same input text `["Do", "n't", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]`.

Despite, the fact these tokenizers do a reasonable job at segregating such compound words. In general, word-level tokenizers result in large vocabularies given a large copora. For instance, [Transformer XL](https://huggingface.co/docs/transformers/model_doc/transfo-xl) uses space and punctuation tokenization and has a vocabulary size of 267,735! Consequently, as described above this results in a huge embedding matrix increasing the complexity and compute requirements for pre-training a language model. A rule of thumb for a monolingual transformer-based language model should have a vocabulary size approximately around 50,000.

Another downside of the word-level tokenizer is its inability to handle out-of-vocabulary words at test time. This means we don’t have a word embedding for that word and thus cannot process the input sequence. A typical solution entails, that all such occurrences of words are typically mapped to a special `<UNK>` token. Such mapping can potentially result in loss of useful information when processing a text sequence by the language model. Similarly, Word-level tokenization treats different forms of the same word (e.g., “open”, “opened”, “opens”, “opening”, etc) as separate types thus, resulting in separate embeddings for each.

 We will use the following copus for all the tokenizer implementations:
```python
corpus = [
    "The sleek black cat gracefully leaps over the sleeping dog.",
    "A nimble gray squirrel effortlessly vaults across the backyard fence.",
    "In the quiet forest, a small rabbit dashes past the resting hare.",
    "With a swift motion, the agile kangaroo hops over the dozing koala.",
    "A fast and agile cheetah sprints across the savannah, leaving dust in its wake."
]
```
### Whitespace Word-level tokenizer implementation
```python
def whitespace_word_level_tokenizer(sequence: str) -> list[str]:
    return sequence.split() 

result = list(map(whitespace_word_level_tokenizer, corpus))
print(result)
```
<em>Output:</em>
```python
[['The',  'sleek',  'black',  'cat',  'gracefully',  'leaps',  'over',  'the',  'sleeping',  'dog.'], ['A',  'nimble',  'gray',  'squirrel',  'effortlessly',  'vaults',  'across',  'the',  'backyard',  'fence.'], ['In',  'the',  'quiet',  'forest,',  'a',  'small',  'rabbit',  'dashes',  'past',  'the',  'resting',  'hare.'], ['With',  'a',  'swift',  'motion,',  'the',  'agile',  'kangaroo',  'hops',  'over',  'the',  'dozing',  'koala.'], ['A',  'fast',  'and',  'agile',  'cheetah',  'sprints',  'across',  'the',  'savannah,',  'leaving',  'dust',  'in',  'its',  'wake.']]
```

### Rule-based Word-level tokenizer implementation
```python

import re 
def rule_based_word_level_tokenizer(sequence: str) -> list[str]:
    """The rules include to replace any special characters other than alphanumeric with a whitespace. Then only extract whitespace separated words."""
    preprocessed_sequence = re.sub(r'\W+', ' ', sequence) 
    return re.findall(r'\b\w+\b|[^\w\s]', preprocessed_sequence) 

result = list(map(rule_based_word_level_tokenizer, corpus))
print(result)
```
<em>Output:</em>
```python
[['The', 'sleek', 'black', 'cat', 'gracefully', 'leaps', 'over', 'the', 'sleeping', 'dog'], ['A', 'nimble', 'gray', 'squirrel', 'effortlessly', 'vaults', 'across', 'the', 'backyard', 'fence'], ['In', 'the', 'quiet', 'forest', 'a', 'small', 'rabbit', 'dashes', 'past', 'the', 'resting', 'hare'], ['With', 'a', 'swift', 'motion', 'the', 'agile', 'kangaroo', 'hops', 'over', 'the', 'dozing', 'koala'], ['A', 'fast', 'and', 'agile', 'cheetah', 'sprints', 'across', 'the', 'savannah', 'leaving', 'dust', 'in', 'its', 'wake']]
```

Above were some toy examples of world-level tokenizers. Additional steps include creating a vocabulary consisting of set of tokens post-tokenization. Thereafter, language models learn a unique embedding per vocabulary item encampasulating the semantic meaning of the token. However, as discussed above word-level tokenization has inherent limitations including large resulting vocabularies, inability to deal with out-of-vocabulary tokens at test time etc.
## Character-level Tokenization

Character-level tokenization alleviates the out-of-vocabulary problem of the word-level tokenization. It solves this by building the vocabulary based on all possible single characters possible for a given natural language. Hence, you can always form any word by putting together individual characters from the vocabulary even when you have misspellings in the input text. Additionally, the vocabulary size becomes much more smaller, for instance, for English language you can potentially have 170,000 unique words. Whereby, to represent the same words you would only require 256 characters. Hence, drastically reducing vocabulary size and computational complexity.

While character-based tokenization provides an elegant solution for gracefully handling out-of-vocabulary terms. Character-based tokenization present its own set of challenges such as characters don't hold more contextual information than individual words. The resulting tokenized sequences are much longer than word-level tokenization due to the fact that each character in your input is a distinct token. Resultantly, the language model is required to deal with longer input contexts. Furthermore, to learn semantic representations at word-level you need to explicitly pool information for each word within the language model. Let's have quick look at a naive character-level tokenizer implementation below.

### Character-level tokenizer implementation
```python

def character_level_tokenizer(sequence: str) -> list[str]:
  return list(sequence)

result = list(map(character_level_tokenizer, corpus))
print(result)
```
<em>Output:</em>
```python
[['T', 'h', 'e', ' ', 's', 'l', 'e', 'e', 'k', ' ', 'b', 'l', 'a', 'c', 'k', ' ', 'c', 'a', 't', ' ', 'g', 'r', 'a', 'c', 'e', 'f', 'u', 'l', 'l', 'y', ' ', 'l', 'e', 'a', 'p', 's', ' ', 'o', 'v', 'e', 'r', ' ', 't', 'h', 'e', ' ', 's', 'l', 'e', 'e', 'p', 'i', 'n', 'g', ' ', 'd', 'o', 'g', '.'], ['A', ' ', 'n', 'i', 'm', 'b', 'l', 'e', ' ', 'g', 'r', 'a', 'y', ' ', 's', 'q', 'u', 'i', 'r', 'r', 'e', 'l', ' ', 'e', 'f', 'f', 'o', 'r', 't', 'l', 'e', 's', 's', 'l', 'y', ' ', 'v', 'a', 'u', 'l', 't', 's', ' ', 'a', 'c', 'r', 'o', 's', 's', ' ', 't', 'h', 'e', ' ', 'b', 'a', 'c', 'k', 'y', 'a', 'r', 'd', ' ', 'f', 'e', 'n', 'c', 'e', '.'], ['I', 'n', ' ', 't', 'h', 'e', ' ', 'q', 'u', 'i', 'e', 't', ' ', 'f', 'o', 'r', 'e', 's', 't', ',', ' ', 'a', ' ', 's', 'm', 'a', 'l', 'l', ' ', 'r', 'a', 'b', 'b', 'i', 't', ' ', 'd', 'a', 's', 'h', 'e', 's', ' ', 'p', 'a', 's', 't', ' ', 't', 'h', 'e', ' ', 'r', 'e', 's', 't', 'i', 'n', 'g', ' ', 'h', 'a', 'r', 'e', '.'], ['W', 'i', 't', 'h', ' ', 'a', ' ', 's', 'w', 'i', 'f', 't', ' ', 'm', 'o', 't', 'i', 'o', 'n', ',', ' ', 't', 'h', 'e', ' ', 'a', 'g', 'i', 'l', 'e', ' ', 'k', 'a', 'n', 'g', 'a', 'r', 'o', 'o', ' ', 'h', 'o', 'p', 's', ' ', 'o', 'v', 'e', 'r', ' ', 't', 'h', 'e', ' ', 'd', 'o', 'z', 'i', 'n', 'g', ' ', 'k', 'o', 'a', 'l', 'a', '.'], ['A', ' ', 'f', 'a', 's', 't', ' ', 'a', 'n', 'd', ' ', 'a', 'g', 'i', 'l', 'e', ' ', 'c', 'h', 'e', 'e', 't', 'a', 'h', ' ', 's', 'p', 'r', 'i', 'n', 't', 's', ' ', 'a', 'c', 'r', 'o', 's', 's', ' ', 't', 'h', 'e', ' ', 's', 'a', 'v', 'a', 'n', 'n', 'a', 'h', ',', ' ', 'l', 'e', 'a', 'v', 'i', 'n', 'g', ' ', 'd', 'u', 's', 't', ' ', 'i', 'n', ' ', 'i', 't', 's', ' ', 'w', 'a', 'k', 'e', '.']]
```

## Subword Tokenization
To get the best of both worlds, transformer-based language models leverage subword tokenization technique which combines both character-level and word-level tokenization mechanisms. Interestingly, the original idea was developed for machine translation by Sennrich et al., ACL 2016. 

<blockquote class="blockstyle" style="color:#00CC8F;">
 <span class="triangle">💡</span> "The main motivation behind this paper is that the translation of some words is transparent in that they are translatable by a competent 
translator even if they are novel to him or her, based on a translation of known subword units such as morphemes or phonemes."
</blockquote>

The key principle of subword tokenization entails applying word-level tokenization to more frequent terms. Whereby, the rather rare terms are decomposed into smaller frequent terms. For instance, the word `annoyingly` can potentially be decomposed into two words `annoying` and `ly`. Hence, this keeps in check the input tokenized sequence length (not decomposing frequent terms), whilst also providing means to handle out-of-vocbulary terms (decomposition of rare terms).

Concretely, the subword toeknization has multiple implementation flavors. Below we will go over the most commonplace flavors used by some of the mainstream language models.

### Byte-Pair (BPE) Tokenization
BPE is used by a lot of Transformer models, including GPT, GPT-2, RoBERTa, BART, and DeBERTa. BPE tokenization involves a training step at the beginning. Unlike the machine learning model the training process entails computing statistical measures to construct a meaningful and robust vocabulary whilst also applying pre-processing (normalization and pre-tokenization) and post-processing steps. For instance, for the toy corpus below the base vocabulary will consist of {"b", "g", "h", "n", "p", "s", "u"}. 

```python
"hug", "pug", "pun", "bun", "hugs"
```

For real-world cases, the base vocabulary will include all Unicode characters at the beginning. In that case, if the tokenizer encounters a character that was not a member of training corpus, it will be mapped to unknown token. This is the reason behind why most NLP models struggle with anlyzing sequences with emojis present.



<blockquote class="blockstyle" style="color:#00CC8F;">
 <span class="triangle">💡</span> The GPT-2 and Roberta tokenizers directly use byte-level representations of all characters as the base vocbulary with size 256. Thereby, this initial trick encompasses all possible character and allows to avoid out-of-vocabulary situation. This trick is called byte-level BPE.
</blockquote>

<strong>Step 0:</strong> <br/>
After forming the base vocabulary, BPE tokenizer looks at each pair of consecutive tokens and concatenates the most frequent token-pair token and adds the concatenated token-pair to base vocabulary and replaces all the consecutive occurences of both tokens with their concatenated version. Here, `+` indicates the token-boundary based on the current state of vocabulary. Since, at the beginning we only have characters in the base vocabulary thus, here's how things would look like at the beginning.
<table>
  <tr>
      <th>Word</th>
      <th>Frequency</th>
  </tr>
  <tr>
      <td>h+u+g</td>
      <td>10</td>
  </tr>
  <tr>
      <td>p+u+g</td>
      <td>5</td>
  </tr>
  <tr>
      <td>p+u+n</td>
      <td>12</td>
  </tr>
  <tr>
      <td>b+u+n</td>
      <td>4</td>
  </tr>
  <tr>
      <td>h+u+g+s</td>
      <td>5</td>
  </tr>
</table>
<h4>Vocbulary: {"b", "g", "h", "n", "p", "s", "u"}</h4>

<strong>Step 1:</strong>
After the first pass we can observe the token-pair `u+g` occurs 20 times in the entire corpus, hence, we concatenate and replace in corpus, add to the vocabulary.
<table>
  <tr>
      <th>Word</th>
      <th>Frequency</th>
  </tr>
  <tr>
      <td>h+ug</td>
      <td>10</td>
  </tr>
  <tr>
      <td>p+ug</td>
      <td>5</td>
  </tr>
  <tr>
      <td>p+u+n</td>
      <td>12</td>
  </tr>
  <tr>
      <td>b+u+n</td>
      <td>4</td>
  </tr>
  <tr>
      <td>h+ug+s</td>
      <td>5</td>
  </tr>
</table>
<h4>Vocbulary: {"b", "g", "h", "n", "p", "s", "u", "ug"}</h4>

<strong>Step 2:</strong> <br/>
In this iteration we find that `u+n` has the highest frequecy of 16.
<table>
  <tr>
      <th>Word</th>
      <th>Frequency</th>
  </tr>
  <tr>
      <td>h+ug</td>
      <td>10</td>
  </tr>
  <tr>
      <td>p+ug</td>
      <td>5</td>
  </tr>
  <tr>
      <td>p+un</td>
      <td>12</td>
  </tr>
  <tr>
      <td>b+un</td>
      <td>4</td>
  </tr>
  <tr>
      <td>h+ug+s</td>
      <td>5</td>
  </tr>
</table>
<h4>Vocbulary: {"b", "g", "h", "n", "p", "s", "u", "ug", "un"}</h4>

<strong>Step 3:</strong> <br/>
In this iteration we find that `h+ug` has the highest frequecy of 15.
<table>
  <tr>
      <th>Word</th>
      <th>Frequency</th>
  </tr>
  <tr>
      <td>hug</td>
      <td>10</td>
  </tr>
  <tr>
      <td>p+ug</td>
      <td>5</td>
  </tr>
  <tr>
      <td>p+un</td>
      <td>12</td>
  </tr>
  <tr>
      <td>b+un</td>
      <td>4</td>
  </tr>
  <tr>
      <td>hug+s</td>
      <td>5</td>
  </tr>
</table>
<h4>Vocbulary: {"b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"}</h4>

The above process is repeated again until a certain criteria (desired vocabulary size is reached) is met or till fixed number of steps.

### WordPiece Toekenization

WordPiece has also been reused in quite a few Transformer models based on BERT, such as DistilBERT, MobileBERT, Funnel Transformers, and MPNET. WordPiece tokenizer is quite similar to the BPE tokenizer while having two distinctive differences: 
- Intermediate characters in have the prepended prefix `##`. 
- Merge of each token-pair is based on greater likelihood a given token-pair is likely to appear in a sequence in the corpus. In other words the algorithm prioritizes the merging of a token-pair where the individual parts are less frequent in the vocabulary. More formally given by for `i` token-pair with individual `j` and `k` as tokens: <br/>

$$score_i = \frac{count(j,k)}{count(j)*count(k)}$$

<strong>Step 0:</strong> <br/>
Hence, as per the above distictions the previously mentioned corpus and the base vocubalary will look as follows at the beginning of the tokenizer training process.

<table>
  <tr>
      <th>Word</th>
      <th>Frequency</th>
  </tr>
  <tr>
      <td>h+##u+##g</td>
      <td>10</td>
  </tr>
  <tr>
      <td>p+##u+##g</td>
      <td>5</td>
  </tr>
  <tr>
      <td>p+##u+##n</td>
      <td>12</td>
  </tr>
  <tr>
      <td>b+##u+##n</td>
      <td>4</td>
  </tr>
  <tr>
      <td>h+##u+##g+##s</td>
      <td>5</td>
  </tr>
</table>
<h4>Vocbulary: {"b", "h", "p", "##g", "##n", "##s", "##u"}</h4>

<strong>Step 1:</strong> <br/>
The highest score after the first iteration will go to the token-pair `##g+##s` with the highest score being `1/20`. Whereby, all other token-pairs have `##u` in them which deflates the scores of all other token-pairs by the factor `1/36`.

<table>
  <tr>
      <th>Word</th>
      <th>Frequency</th>
  </tr>
  <tr>
      <td>h+##u+##g</td>
      <td>10</td>
  </tr>
  <tr>
      <td>p+##u+##g</td>
      <td>5</td>
  </tr>
  <tr>
      <td>p+##u+##n</td>
      <td>12</td>
  </tr>
  <tr>
      <td>b+##u+##n</td>
      <td>4</td>
  </tr>
  <tr>
      <td>h+##u+##gs</td>
      <td>5</td>
  </tr>
</table>
<h4>Vocbulary: {"b", "h", "p", "##g", "##n", "##s", "##u", "##gs"}</h4>

<strong>Step 2:</strong> <br/>
At this point all the token-pairs have the identical scores because all of them include `##u` as one part of the pair. Let's say we merge `h+##u` first.

<table>
  <tr>
      <th>Word</th>
      <th>Frequency</th>
  </tr>
  <tr>
      <td>hu+##g</td>
      <td>10</td>
  </tr>
  <tr>
      <td>p+##u+##g</td>
      <td>5</td>
  </tr>
  <tr>
      <td>p+##u+##n</td>
      <td>12</td>
  </tr>
  <tr>
      <td>b+##u+##n</td>
      <td>4</td>
  </tr>
  <tr>
      <td>hu+##gs</td>
      <td>5</td>
  </tr>
</table>
<h4>Vocbulary: {"b", "h", "p", "##g", "##n", "##s", "##u", "##gs", "hu"}</h4>

<strong>Step 3:</strong> <br/>
After this iteration we observe the maximum score for the token pair `hu+##gs` with score being `1/15`.

<table>
  <tr>
      <th>Word</th>
      <th>Frequency</th>
  </tr>
  <tr>
      <td>hu+##g</td>
      <td>10</td>
  </tr>
  <tr>
      <td>p+##u+##g</td>
      <td>5</td>
  </tr>
  <tr>
      <td>p+##u+##n</td>
      <td>12</td>
  </tr>
  <tr>
      <td>b+##u+##n</td>
      <td>4</td>
  </tr>
  <tr>
      <td>hugs</td>
      <td>5</td>
  </tr>
</table>
<h4>Vocbulary: {"b", "h", "p", "##g", "##n", "##s", "##u", "##gs", "hu", "hugs"}</h4>

Similarly to the BPE tokenization, above process is repeated again until a certain criteria (desired vocabulary size is reached) is met or till fixed number of steps.

### Unigram Tokenization
<blockquote class="blockstyle" style="color:#00CC8F;">
 <span class="triangle">💡</span> Unigram is not used directly for any of the models in the transformers, but it’s used in conjunction with SentencePiece.
</blockquote>

$$L = -\sum_{i=1}^{N} \log \left(\sum_{x \in S(x_i)} p(x)\right)
$$


### SentencePiece Tokenization
Langauge models using SentencePiece are ALBERT, XLNet, Marian, and T5. The issue with above described tokenization algorithms lies in the assumption that input text is delimited by spaces, which doesn't hold true for all languages. A potential remedy involves employing language-specific pre-tokenizers, such as XLM's pre-tokenizer for Chinese, Japanese, and Thai. For a more comprehensive solution, [SentencePiece](https://arxiv.org/pdf/1808.06226.pdf) (Kudo et al., 2018) offers a language-independent approach. It treats input as a raw stream, encompassing spaces within the vocbulary set, and utilizes the BPE or unigram algorithm to construct an appropriate vocabulary.

<blockquote class="blockstyle" style="color:#00CC8F;">
 <span class="triangle">💡</span> All transformers models in the library that use SentencePiece use it in combination with unigram.
</blockquote>
