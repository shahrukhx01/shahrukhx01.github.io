<link href="styles.css" rel="stylesheet"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>

# Exact Matchers: A Primer for Near-Exact String Matching 

<p style="opacity: 0.5;">25 December, 2023</p>
<hr>

- [Introduction](#introduction)
- [Regex-based Exact String Matching](#fuzzy-string-matching)
- [Neural and Heuritic-based Fuzzy String Matching](#fuzzy-string-matching)
- [Jaccard-based String Matching](#jaccard-based-string-matching)
- [w-Shingling-based String Matching](#w-shingling-based-string-matching)
- [Hamming-distance-based String Matching](#hamming-distance-based-string-matching)
- [TF-IDF-based String Matching](#tf-idf-based-string-matching)
- [BM25-based String Matching](#bm25-based-string-matching)
- [Locality Sensitivity Hashing for String Matching at Scale](#locality-sensitive-hashing-for-optimized-matching)
- [Towards Optimized Hybrid String Matching](#towards-hybrid-string-matchers)
- [Summary](#summary)

## Introduction

## Regex-based Exact String Matching
Regex-based string matchers are among the trivial variants of matchers. It mainly entails application of a exact query matching regular expression and looking for it in the target text. Below is an example of such a matcher:
```python
import re

target_string = "Search for anything in my data content."
queries = ["Data", "dataset"]

for query in queries:
    hits = re.search(re.compile(r"(^|[^\w]){}([^\w]|$)".format(query.strip().lower()), re.IGNORECASE), target_string)

    if hits:
        print(f"Match found: {hits.group().strip()}")

# Output:
# Match found: data
```

## Neural and Heuristic-based Fuzzy String Matching
## Heuristic-based Fuzzy String Matching
Fuzzy string matching primarily inv
## w-Shingling-based String Matching