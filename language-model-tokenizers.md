<link href="styles.css" rel="stylesheet"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>

# LLM Tokenizers: Understanding Text Tokenizers for Transformer-based Language Models

<p style="opacity: 0.5;">25 December, 2023</p>
<hr>

- [Introduction](#introduction)

## Introduction

As it is established practice that Computers represent the data digitally in the binary format. Similarly, Large Language Models (LLMs) do not directly process the text provided by the users in the shape of a prompt. The user prompts are passed through a multi-stage pre-processing pipeline prior to being fed to a LLM. Furthermore, you may have came across the "tokens" while navigating the pricing pages of the popular commercial LLM service providers.


In this post, we will expolore various tokenization mechanisms. Including the ones used by the main stream LLMs including LLAMA, GPT, BERT etc. Alongside the simulated tokenization dry runs, we will also go over the implementation of the such tokenization algorithms alongside.
