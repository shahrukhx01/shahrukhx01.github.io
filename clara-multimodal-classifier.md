<link href="styles.css" rel="stylesheet"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>

# A Deep Dive into Clara for Multimodal (Speech & Language) Modeling

<p style="opacity: 0.5;">23 November, 2023</p>
<hr>

- [Introduction](#introduction)
- [Dataset Curation](#dataset-curation)
- [Method](#method)
- [Loss Function](#loss-function)
- [Implementing a Clara-based Multimodal Classifier](#implementing-a-clara-based-multimodal-classifier)
- [Results](#results)
- [Next Steps](#next-steps)

## Introduction

There has been a resurgence in interest towards multimodal models as the the AI research pushes toward Artificial General Intelligence (AGI). Furthermore, due to the unification of the neural architecture, more succinctly put the transformer architecture has been the workhorse behind this renaissance. Here, we will take a closer look at both the architecture and implementation of one such recent approach namely `CLARA: Multilingual Contrastive Learning for
Audio Representation Acquisition` by `Noriy et . al 2023` which entails jointly training a speech and language model for usecase including zeroshot audio classification, audio-based retrieval of text and vice versa.

The key premise of the work hinges on the notion of using self-supervised contrastive loss between the projected hidden representations from unlabelled multilingual speech and language data. The projected representations imply here, that we first pass the audio and text data through their respective encoders. Then a separate projection per modality (Feed-forward Layers) which projects the respective encoder's representation to the joint multimodal latent space. Here, the contrastive loss ensures that same text and audio pairs are pulled closer to each other and vice versa. This work is an extension to the pre-training paradigm introduced the OpenAI's CLIP paper. This can be more concretely inspected in the model's architecture below:

![alt text](/media/clara-multimodal-classifier/architecture.png "Clara Architecture")
Subsequently, once the model has been pre-trained with the aforementioned self-supervised contrastive pre-training objective, the model can be used to perform zero-shot inference by simultaneously passing a prompt `A person talking in a {label} voice` where $$label \in \{ \text{happy, sad, neutral, surprised} \}$$ for emotion recognition task. An example of zero-shot inference is presented below:

![alt text](/media/clara-multimodal-classifier/zeroshot.png "Clara Zeroshot Inference")

Now that we have walked through the high level conceptual overview of the paper, the rest of blogpost is setup as follows. We'd begin by looking the dataset curation for pre-training, go over the key training details including the inner workings of the loss function, finally, we will train a multimodal classifier for emotion recognition on the [Multimodal EmotionLines Dataset (MELD)](https://www.kaggle.com/datasets/zaber666/meld-dataset/data) dataset by using a pre-trained checkpoint of the Clara model.

## Dataset Curation

The authors curate a multimodal data pile composed of recorded audio and text pairs encompassing over 16000 hours of natural language-like speech recordings and also 6000 hours of naturally occuring enviroment sounds. Subsequently, all the training samples are subjected to pre-processing including LAC conversion and resampling to 48kHz. Afterwards, in order to enhance diversity within the training dataset, the authors employ a range of augmentation techniques on the audio signals. These techniques encompass the addition of reverb, clipping, masking, pitch modulation, and the introduction of environmental sounds. This is done to simulate diverse acoustic environments or create distorted audio representations. Formally, the augmentations can be expressed as, given a pair of sound signals $$a_i$$ and $$a_j$$, where, $$a_i$$ is a speech signal and $$a_j$$ corresponds to environmental sound. A scaling constant $$\lambda$$ whose value ranges between 0.01 and 0.8 is used to control the contribution of $$a_j$$ (environment sound) to the $$a_i$$ (speech signal). The corresponding text labels of speech and environment sound are combined using a generative neural language model which generates a more fluent textual representation in reponse to a prompt i.e. `A person saying It’s raining outside, background wind noise`, this can be precisely expressed as follows:

<p style="text-align: center;">$$\hat{a} = \lambda a_i + (1 - \lambda) a_j$$</p>
<p style="text-align: center;">$$\hat{t} = f(p^t + t_i + t_j)$$</p>

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
- $$\log \frac{\exp((z_a^k \cdot z_t^k))\tau_a}{\sum_{i=1}^{N} \exp((z_a^k \cdot z_t^n))\tau_a}$$ depicts the contrastive optimization objective for the audio modality where for each audio datapoint in the minibatch the hidden representations are learned with respect to all text samples based on the similarity in the latent space.
- $$\log \frac{\exp((z_t^k \cdot z_a^k))\tau_t}{\sum_{i=1}^{N} \exp((z_t^k \cdot z_a^n))\tau_t}$$ represents the identical optimization phenomenon albeit for text inputs.

## Implementing a Clara-based Multimodal Classifier

Now having equipped with all the necessary pre-requisites, let's fine-tune the [medium-checkpoint](https://huggingface.co/knoriy/CLARA/blob/main/clara-medium.ckpt) (109 million parameters) Clara for the emotion recognition task. We will begin by download and pre-processing the MELD dataset. The Multimodal EmotionLines Dataset (MELD) was developed by expanding and improving upon the existing EmotionLines dataset. While maintaining the dialogue instances present in EmotionLines, MELD goes a step further by incorporating audio and visual components in addition to text. With over 1400 dialogues and 13000 utterances extracted from the Friends TV series, MELD features contributions from multiple speakers engaged in various conversations. Every utterance within a dialogue has been categorized into one of seven emotions: Anger, Disgust, Sadness, Joy, Neutral, Surprise, and Fear.

The dataset can be downloaded directly from Kaggle [here](https://www.kaggle.com/datasets/zaber666/meld-dataset/data). After downloading the dataset, move the video files from each split to the folder corresponding to the name of the split. Hence, the final directory structure should look as follows:

```
├── meld
│   ├── train
│   │   ├── train_sent_emo.csv
│   │   ├── videofiles
│   │   │   ├── diag0_utt0.mp4
│   ├── dev
│   │   ├── dev_sent_emo.csv
│   │   ├── videofiles
│   │   │   ├── diag1_utt0.mp4
│   ├── test
│   │   ├── test_sent_emo.csv
│   │   ├── videofiles
│   │   │   ├── diag2_utt0.mp4
```

Next we can run the follwing python script on each the `mp4` file to extract the audio track in the `mp3` format and dump that in the `voicefiles` folder, which would be a sibbling to `videofiles`.

```python
import pandas as pd
from moviepy.editor import AudioFileClip


split_names = ["train", "dev", "test"]
for split_name in split_names:
    base_path = f"meld/{split_name}"
    csv = pd.read_csv(f"{base_path}/{split_name}_sent_emo.csv")
    dataset_split = []
    for idx, row in csv.iterrows():
        """
        There are some files which produce encoding errors. Hence, we skip them.
        Thereby, we would generate a new subset based csv file for each split.
        """
        try:
            dialogue_id = row["Dialogue_ID"]
            utterance_id = row["Utterance_ID"]

            file_path = f"{base_path}/videofiles/dia{dialogue_id}_utt{utterance_id}.mp4"
            mp3_file_path = f"{base_path}/voicefiles/dia{dialogue_id}_utt{utterance_id}.mp3"

            file_to_convert = AudioFileClip(file_path)
            file_to_convert.write_audiofile(mp3_file_path)
            file_to_convert.close()
            dataset_split.append(dict(row))
        except:
            continue
    pd.DataFrame(dataset_split).to_csv(f"{base_path}/{split_name}.csv", index=False, encoding='utf-8-sig')

```

Finally, below is how the dataset would look like:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import imageio
from moviepy.editor import AudioFileClip
import tempfile

import torchaudio
import soundfile as sf


from clara_mm_classifiers.utils.tokenizer import Tokenizer
from clara_mm_classifiers.utils.data_util import get_log_melspec

label_encoder = LabelEncoder()
tokenizer = Tokenizer()

class MELDDataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50

    def __init__(
        self,
        path: Path = Path("data/meld"),
        sample_rate: int = 16000,
        split_name: str = "train",
    ):
        # Load CSV & initialize all torchaudio.transforms:
        self.csv = pd.read_csv(path / f"{split_name}/{split_name}.csv")
        self.audio_files_path = path / f"{split_name}/voicefiles"
        self.sample_rate = sample_rate
        self.split_name = split_name

        # create transformed numerical label
        if split_name == "train":
            label_encoder.fit(self.csv.Emotion.values.tolist())
        self.csv["label"] = label_encoder.transform(self.csv.Emotion.values.tolist())

    def __getitem__(self, index):
        current_row = self.csv.iloc[index]
        dialogue_id = current_row["Dialogue_ID"]
        utterance_id = current_row["Utterance_ID"]
        label = current_row["label"]

        file_path = f"{self.audio_files_path}/dia{dialogue_id}_utt{utterance_id}.mp3"

        audio, _ = sf.read(file_path)
        audio = audio.astype(float)
        audio = audio[:, 0]
        text = current_row["Utterance"]
        return audio, text, label


    def __len__(self):
        # Returns length
        return len(self.csv)

# `collate_fn` defines how to stack multiple data points in a batch
def collate_fn(batch):
    audios = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    texts = [torch.tensor(tokenizer.encode(text, lang="en")) for text in texts]

    mels = [get_log_melspec(np.array(a), 16000) for a in audios]
    mel_lengths = [mel.shape[0] for mel in mels]
    mel_lengths = torch.tensor(mel_lengths)

    text_lengths = [text.size(0) for text in texts]
    text_lengths = torch.tensor(text_lengths)

    mels = pad_sequence(mels).squeeze(-1).permute(1,2,0).contiguous()
    texts = pad_sequence(texts).T.contiguous()
    labels = torch.LongTensor(labels)

    return labels, mels, texts, text_lengths, mel_lengths

```

Now let's move on to the modelling the classifier, here, we would add additional fully connected layers on top of the pre-trained Clara model which has both bespoke audio and text encoders followed by a projection head. Importantly, the [original implementation](https://github.com/knoriy/CLARA/tree/master/clara) of the paper is based on Pytorch Lightning hence, we would also be using that to incorporate our linear classifier. Lastly, since our task is to classify a tuple of `(audio, text)` to one of the following labels (Anger, Disgust, Sadness, Joy, Neutral, Surprise, and Fear). Hence, we use Cross Entropy loss as the choice of loss function.

```python
from typing import Literal

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from clara_mm_classifiers.models.clara import PLCLARA
from clara_mm_classifiers.models.encoders.layer_modules import MLPLayers
from clara_mm_classifiers.utils.accuracy import accuracy
from clara_mm_classifiers.utils.modeling_utils import get_optimiser


class CLARAAudioMultimodalLinearProbe(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        clara_checkpoint_path: str,
        clara_map_location: str = "cuda",
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        learning_rate_patience: int = 10,
        LR_sheduler_T_max: int = 40,
        LR_sheduler_warmup_steps: int = 5,
        LR_sheduler_min_lr: float = 0.0,
        LR_sheduler_decay: float = 1.0,
        lr_interval: Literal["epoch", "step"] = "epoch",
        *args,
        **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.feature_extractor = PLCLARA.load_from_checkpoint(
            clara_checkpoint_path, map_location=clara_map_location
        )
        self.feature_extractor.freeze()

        self.classifier = MLPLayers(
            [2 * self.feature_extractor._hparams.output_dim, 512, 128, num_classes],
            dropout=dropout,
        )

    def forward(self, text: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:

        text_features = self.feature_extractor.encode_text(text)
        audio_features = self.feature_extractor.encode_audio(audio)

        # Projection
        text_features = self.feature_extractor.model.text_transform(text_features)
        audio_features = self.feature_extractor.model.audio_transform(audio_features)

        text_features = F.normalize(text_features, dim=-1)
        audio_features = F.normalize(audio_features, dim=-1)

        return self.classifier(torch.cat((audio_features, text_features), 1))

    def training_step(self, batch, batch_idx):
        _, loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"validation_accuracy": acc, "val_loss": loss}
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_accuracy": acc, "test_loss": loss}
        self.log_dict(metrics)

    def _shared_eval_step(self, batch, batch_idx):
        labels, mels, texts, _, _ = batch
        y_hat = self(texts, mels).squeeze()

        loss = F.cross_entropy(y_hat, labels)
        acc = accuracy(y_hat, labels)[0] / labels.size(0)

        return y_hat, loss, acc

    def configure_optimizers(self):
        return get_optimiser(self)

```

Finally, let's add our trainer for training, validation and evalaution post-training. We would instantiate our trainer with MLFlow for experiment tracking and furthermore, we would like to stop training if the validation accuracy does not improve over multiple epochs. Thereby, we also pass the `EarlyStopping` callback to the trainer, which essentially monitors validation accuracy.

```python
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from clara_mm_classifiers.datasets.meld_dataset import MELDDataset, collate_fn
from clara_mm_classifiers.models.clara_multimodal_linear_probe import CLARAAudioMultimodalLinearProbe


if __name__ == "__main__":
    # setup paths
    path = Path("meld/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pre-trained base model weights and initialize classification layers
    model_path = "clara-experiments/clara-medium.ckpt"
    model = CLARAAudioMultimodalLinearProbe(num_classes=7, clara_checkpoint_path=model_path, clara_map_location=device)

    # configure datasets and datloaders
    batch_size = 64
    num_dataloader_workers = 16
    train_data = MELDDataset(path=path)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_dataloader_workers
    )
    dev_data = MELDDataset(path=path, split_name="dev")
    dev_loader = torch.utils.data.DataLoader(
        dev_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_dataloader_workers
    )

    test_data = MELDDataset(path=path, split_name="test")
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_dataloader_workers
    )

    # configure trainer and start training
    mlf_logger = MLFlowLogger(experiment_name="clara-multimodal-classifier", tracking_uri="http://localhost:5000")
    early_stop_callback = EarlyStopping(monitor="validation_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")

    trainer = pl.Trainer(max_epochs=20, logger=mlf_logger, accelerator="cuda", precision=16, callbacks=[early_stop_callback])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=train_loader)

    # evaluate trained model
    trainer.test(model=model, dataloaders=test_loader)
```

We can then initiate the fine-tuning process for the length of `20` epochs. The choice of number of epoch is based on the fine-tuning experiments from the paper itself. Where, authors argue with the training set to `20` epochs, the model is less likely to overfit and hence, retain high-fidelity latent space learned during pre-training whilst, adopting sufficiently enough to the downstream task.

## Results

Below is the visual summary of how the training went, we first have training and validation loss curves plotted over the training steps:
![alt text](/media/clara-multimodal-classifier/train_val_loss.png "Training & Validation Loss")
It is interesting to observe that both the training and validation losses were converging prior to the conclusion of the training. This implies that training for further epochs can potentially be promising.

Furthermore, we also have a plot showing evolution of validation accuracy over the 20 epochs, as the validation accuracy never stagnated during training, hence, early stopping did not stop training in the middle.
![alt text](/media/clara-multimodal-classifier/val_accuracy.png "Validation Accuracy")

Lastly, on the test set we get an accuracy score of $$~49\%$$, it is important to note that the state of the art model on MELD dataset achieves $$~69\%$$. Hence, it is impressive to get a reasonable accuracy score only after fine-tuning (classifier layers only) for 20 epochs.

## Summary and Next Steps

In conclusion, we have discussed one of the recent appraoch for pre-training a multimodal (speech and language) model named CLARA. Furthermore, we applied the pre-trained medium checkpoint to MELD dataset by fine-tuning the model. The fine-tuning process involved adding few fully connected layers next to the projection head of the model.

As concrete next steps, the performance of the above classifier can be even further improved by hyperparameter tuning and/or unfreezing both the decoders, as currently we do not update the encoder parameters during fine-tuning. Lastly, if you have multiple similar classification tasks at hand, multi-task fine-tuning can also potentially add a performance boost to the downstream classifier and is more data-efficient in terms of training samples.
