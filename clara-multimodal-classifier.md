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

The key premise of the work hinges on the notion of using self-supervised contrastive loss between the projected hidden representations from unlabelled multilingual speech and language data. The projected representations imply here, that we first pass the audio and text data through their respective encoders. Then a separate projection per modality (Feed-forward Layers) which projects the respective encoder's representation to the joint multimodal latent space. Here, the contrastive loss ensures that same text and audio pairs are pulled closer to each other and vice versa. This work is an extension to the pre-training paradigm introduced the OpenAI's CLIP paper. This can be more concretely inspected in the model's architecture below:

![alt text](/media/clara-multimodal-classifier/architecture.png "Clara Architecture")
Subsequently, once the model has been pre-trained with the aforementioned self-supervised contrastive pre-training objective, the model can be used to perform zero-shot inference by simultaneously passing a prompt `A person talking in a {label} voice` where $$label \in \{ \text{happy, sad, neutral, surprised} \}$$ for emotion recognition task. An example of zero-shot inference is presented below:

![alt text](/media/clara-multimodal-classifier/zeroshot.png "Clara Zeroshot Inference")

Now that we have walked through the high level conceptual overview of the paper, the rest of blogpost is setup as follows. We'd begin by looking the dataset curation for pre-training, go over the key training details including the inner workings of the loss function, finally, we will train a multimodal classifier for emotion recognition on the [Multimodal EmotionLines Dataset (MELD)](https://www.kaggle.com/datasets/zaber666/meld-dataset/data) dataset by using a pre-trained checkpoint of the Clara model.

## Dataset Curation
The authors curate a multimodal data pile composed of recorded audio and text pairs encompassing over 16000 hours of natural language-like speech recordings and also 6000 hours of naturally occuring enviroment sounds. Subsequently, all the training samples are subjected to pre-processing including LAC conversion and resampling to 48kHz. Afterwards, in order to enhance diversity within the training dataset, the authors employ a range of augmentation techniques on the audio signals. These techniques encompass the addition of reverb, clipping, masking, pitch modulation, and the introduction of environmental sounds. This is done to simulate diverse acoustic environments or create distorted audio representations. Formally, the augmentations can be expressed as, given a pair of sound signals $$a_i$$ and $$a_j$$, where, $$a_i$$ is a speech signal and $$a_j$$ corresponds to environmental sound. A scaling constant $$\lambda$$ whose value ranges between  0.01 and 0.8 is used to control the contribution of $$a_j$$ (environment sound) to the $$a_i$$ (speech signal). The corresponding text labels of speech and environment sound are combined using a generative neural language model which generates a more fluent textual representation in reponse to a prompt i.e. `A person saying It’s raining outside, background wind noise`, this can be precisely expressed as follows:

$$\hat{a} = \lambda a_i + (1 - \lambda) a_j$$ <br/>
$$\hat{t} = f(p^t + t_i + t_j)$$

To enhance model's multilingual capabilities the authors also additionally translate the labels of environment sounds to further languages. This is feasible since environment sounds are not based on linguistic information, hence, don't require any amendments.
## Method
The Clara architecture primarily relies on a contrastive training objective which enables maximization of similarity between texts which correspond to the correct audio, otherwise, their mutual similarity in the latent space is minimized towards zero. The neural network's architecture is constituted by two encoders $$f_a$$ and $$f_t$$ that take log melspectograms and tokenized texts extracted from input audio and text respectively. The audio encoder $$f_a$$ produces hidden representation $$z_a$$, whereby, the text encoder $$f_t$$ learns the hidden representation $$z_t$$. Finally, both hidden representations are pass through the projection head $$g(.)$$ which projects both latent representations into a shared latent space for audio and textual representations. The projection enables the contrastive objective to perform optimization over the audio and text pairs in the shared latent space.

Additionally, both the encoders are based on the infamous transformer architecture and use consine positional embeddings to encoder temporal aspect into hidden representations. For text encoder $$f_t$$, the authors additionally employ flash attention for optimized application of multi-headed attention mechanism. Whereby, the audio encoder $$f_a$$, is based on the Perceiver head with some modifications including the use of GELU activations for key and values instead of Softmax, and introduction a convolution head added at the beginning of the $$f_a$$ encoder. Furthermore, for the $$f_a$$ an additional latent vector is learned to process outputs with constant number of operations. In other words, the number of parameters are not linearly scaled if the length of the audio changes.

### Loss Function
The author extend the CLIP loss function to learn the aligned shared representations of audio and text. The authors argue that the proposed extension allows for parallization of multimodal alignment process and robust learning capabilities encompassing nuannces found in both modalities. Formally, the loss function is expressed as follows:

$$L = \frac{1}{N} \sum_{k=1}^{N} \left[ \log \frac{\exp((z_a^k \cdot z_t^k))\tau_a}{\sum_{i=1}^{N} \exp((z_a^k \cdot z_t^n))\tau_a} + \log \frac{\exp((z_t^k \cdot z_a^k))\tau_t}{\sum_{i=1}^{N} \exp((z_t^k \cdot z_a^n))\tau_t} \right]$$

Now, let's break down the loss function component by component:

- $$\tau_a$$ and $$\tau_t$$ are learnable temperature parameters. They predominantly control the entropy in the model's output distribution for audio and text modalities repectively.
- $$N$$ corresponds to the minibatch size.
- $$z_a^k$$ and $$z_t^k$$ correspond to the kth audio and text pair's hidden representations after the projection.
- $$\log \frac{\exp((z_a^k \cdot z_t^k))\tau_a}{\sum_{i=1}^{N} \exp((z_a^k \cdot z_t^n))\tau_a}$$ depicts the contrastive optimization objective for the audio modality where for each audio datapoint in the minibatch the hidden representations are learned wrt to the audio sample based on the similarity in the latent space.
- $$\log \frac{\exp((z_t^k \cdot z_a^k))\tau_t}{\sum_{i=1}^{N} \exp((z_t^k \cdot z_a^n))\tau_t}$$ represents the identical optimization phenomenon albeit for text inputs.

## Implementing a Clara-based Multimodal Classifier 



[^acceptance]: <Add foot notes>