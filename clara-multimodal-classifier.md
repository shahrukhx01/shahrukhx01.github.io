<link href="styles.css" rel="stylesheet"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>

# A Deep Dive into Clara for Multimodal (Speech & Language) Modeling
<p style="opacity: 0.5;">23 November, 2023</p>
<hr>

- [Introduction](#introduction)
- [Dataset Curation](#dataset-curation)
- [Method](#dataset-curation)
- [Implementing a Clara-based Multimodal Classifier](#implementing-a-clara-based-multimodal-classifier)


## Introduction
There has been a resurgence in interest towards multimodal models as the the AI research pushes toward Artificial General Intelligence (AGI). Furthermore, due to the unification of the neural architecture, more succinctly put the transformer architecture has been the workhorse behind this renaissance. Here, we will take a closer look at both the architecture and implementation of one such recent approach namely ```CLARA: Multilingual Contrastive Learning for
Audio Representation Acquisition``` by `Noriy et . al 2023` which entails jointly training a speech and language model for usecase including zeroshot audio classification, audio-based retrieval of text and vice versa. 

The key premise of the work hinges on the notion of using self-supersvised contrastive loss between the projected hidden representations from unlabelled multilingual speech and language data. The projected representations imply here, that we first pass the audio and text data through their respective encoders. Then a separate projection per modality (Feed-forward Layers) which projects the respective encoder's representation to the joint multimodal latent space. Here, the contrastive loss ensures that same text and audio pairs are pulled closer to each other and vice versa. This work is an extension to the pre-training paradigm introduced the OpenAI's CLIP paper. This can be more concretely inspected in the model's architecture below:

![alt text](/media/clara-multimodal-classifier/architecture.png "Clara Architecture")
Subsequently, once the model has been pre-trained with the aforementioned self-supervised contrastive pre-training objective, 

Now that we have walked through the high level conceptual overview of the paper, the rest of blogpost is setup as follows. We'd begin by looking the dataset curation for pre-training, go over the key training details including the inner workings of the loss function, finally, we will train a multimodal classifier for emotion recognition on the [Multimodal EmotionLines Dataset (MELD)](https://www.kaggle.com/datasets/zaber666/meld-dataset/data) dataset by using a pre-trained checkpoint of the Clara model.

## Dataset Curation

## Implementing a Clara-based Multimodal Classifier 



[^acceptance]: <Add foot notes>