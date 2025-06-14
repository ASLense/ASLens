
# ASLens - Sign Language to Text Conversion

ASLens is a deep learning project that converts sign language gestures into text, using deep recurrent neural networks.

  

# Overview

**ASLens** is an assistive tool designed to promote the inclusion of people with hearing loss by translating sign language gestures into written text. This project leverages deep learning techniques to convert sign language video data into meaningful textual output.

The primary goal of ASLens is to translate sign language gestures into readable text using Recurrent Neural Networks (RNNs)

-   **Dataset**: The system is trained on the How2Sign dataset, which provides a rich collection of sign language video data.
    
-   **Feature Extraction**: Visual features are extracted from the videos using [MediaPipe](https://mediapipe.dev/), which detects and tracks key points of hand and body movements.
    
-   **Sequence Modeling**:
    
    -   An **RNN Encoder** processes the extracted gestures to understand how the signs change and move over time.
        
    -   A **CharRNN Decoder** then translates these encoded features into coherent text sentences, character by character.

# Dataset

## Step 1: Collecting Data

ASLens uses the **[How2Sign ](https://how2sign.github.io/)** dataset, which consists of video data of sign langauge gestures (ASL language), with the corresponding text translation.

## Step 2: Extracting Hand - Face Keypoints using MediaPipe

To prepare the data for model training, we use **[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)** to extract key landmarks from the **hands** and **face** in each video frame. MediaPipe provides pre-trained models that detect these keypoints, which are crucial for understanding and interpreting sign language gestures.

Due to limited computational resources, we processed the videos at a reduced frame rate of **15 frames per second (fps)** to make the training more efficient.

After detecting the landmarks using our custom `DataExtractor` tool, we selected a subset of the most relevant hand and face keypoints—those carrying the most meaningful information for sign interpretation.  These are:

-   **Hand landmarks**: For each hand, we extract 21 landmarks representing the wrist, palm center, and key points along each finger (including the finger tips and joints).
    
-   **Face landmarks**: For the face, we extract 20 landmarks outlining  the mouth and lips, and 36 landmarks that outline the jawline and forehead region.


The final extracted features from each frame are stored as a tensor. The tensors of each landmark type are concatenated, resulting in a shape of [frames, 98, 3], where 98 corresponds to the combined number of landmarks. The total 98 landmarks include: 42 hand (21×2), 20 lip, and 36 facial points. Each landmark is comprised of 3 coordinates (x, y, z). These tensors are then used as input sequences for the model

![](https://github.com/ASLense/ASLens/blob/main/assets/nl-gif.gif)
![](https://github.com/ASLense/ASLens/blob/main/assets/l-gif.gif)

# Model Architecture
## Experiment 1: Encoder-Decoder Architecture with CharRNN as Decoder
### Step 1: Training CharRNN

Firstly, we train the CharRNN on a large corpus of Wikipedia text to learn character-level dependencies in natural language. We use an LSTM-based architecture with character embeddings to model sequential patterns, followed by a fully connected layer to produce the final character label at each time step. To further improve performance, we incorporate pretrained word embeddings (**[Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)**), allowing the model to benefit not only from character-level context but also from word-level semantics.
<div align="center">

<table>
<tr>
<td>

| Architecture Component | Value         |
|------------------------|---------------|                                                              
| **Model Type**         | LSTM          |
| **Hidden Size**        | 384           |
| **Number of Layers**   | 3             |

</td>

<td>

```mermaid
graph TD
    A[Input] --> B[LSTM<br>384 units]
    B --> C[FC Layer<br>1024 units]
    C --> D[Output]
```
</td> </tr> </table> 
</div>

### Step 2: Encoder-Decoder network
### Encoder architecture

The encoder is the main component of our architecture, responsible for mapping landmark features into the model’s latent space. To achieve this, we use a combination of convolutional layers and recurrent layers. The data first flows through a series of 1D convolutional layers, which capture local temporal patterns in the input sequence. The output of the convolutional layers is then passed through an LSTM, which models longer-range dependencies and computes the final feature representation.


### Encoder Architecture Overview
<div align="center">

| Component               | Configuration                          |
|-------------------------|----------------------------------------|
| **Input Shape**         | `(batch, time, 98, 3)`                | 
| **Conv1D Block**        |                                        |
| → Layer 1               | `Conv1d(3→16, kernel=3, padding=1)`   | 
| → Layer 2               | `Conv1d(16→32, kernel=2, padding=1)`  | 
| → Layer 3               | `Conv1d(32→64, kernel=2, padding=1)`  | 
| **LSTM Block**          |                                        |
| → Hidden Size           | `384`                                  |
| → Layers                | `3`                                    |
| **Output**              |                                         |

</div>

```mermaid

graph LR
A["Input<br>(frames, 98, 3)"]
A --> C["Conv1D<br>3→16 channels<br>kernel=3, pad=1"]
C --> D["Conv1D<br>16→32 channels<br>kernel=2, pad=1"]
D --> E["Conv1D<br>32→64 channels<br>kernel=2, pad=1"]
E --> G["LSTM<br>Hidden Size=384<br>Layers=3"]
G --> H["Output<br>"]
```

---
### Encoder-Decoder Merge
To produce the text output, we pass the landmark sequences through the encoder, which generates a hidden state representation. This hidden state is then passed as the initial hidden state to the CharRNN decoder, which generates the output sequence token by token.
```mermaid

  

graph TD
subgraph Inputs
	A["Landmark sequences<br>(frames, 98, 3)"]
	T["#lt;SOS#gt;How ar"]
	A ~~~ T
end

A --> B["Encoder"]

B -.-> |"Hidden State<br>(h₀, c₀)"|C["CharRNN Decoder"]
T--> P["Tokenizer"]
P -.-> |"Character-Level<br> Encoding"|C
P -.-> |"Word2Vec<br> Embeddings<br>"|C
C --> F["Next Character<br>p(yₜ|yₜ₋₁, h) = 'e'"] -.-> T
style T fill:#e9b116,stroke:none,color:black


```
---
## Experiment 2: Encoder-Decoder Architecture with CharRNN as Decoder
### Encoder architecture

The encoder architecture remains the same as in Experiment 1. It combines 1D convolutional layers for capturing local temporal patterns with LSTM layers for modeling longer-range dependencies and generating the final feature representation.


```mermaid

graph LR
A["Input<br>(frames, 98, 3)"]
A --> C["Conv1D<br>3→16 channels<br>kernel=3, pad=1"]
C --> D["Conv1D<br>16→32 channels<br>kernel=2, pad=1"]
D --> E["Conv1D<br>32→64 channels<br>kernel=2, pad=1"]
E --> G["LSTM<br>Hidden Size=384<br>Layers=3"]
G --> H["Output<br>"]
```
### Encoder-GPT2 Merge
GPT-2 is a large pretrained model trained on a vast corpus of text, and it has already learned strong text dependencies. To integrate our encoder with GPT-2, we use **cross-attention**: we pass the encoder’s hidden state as the context to GPT-2’s cross-attention layers. This lets GPT-2 use encoded sign language features, combining  its pretrained language knowledge with the input landmark representations.
```mermaid

graph TD
subgraph Inputs
	A["Landmark sequences<br>(frames, 98, 3)"]
	T["#lt;SOS#gt;How ar"]
	A ~~~ T
end

A --> B["Encoder"]

B -.-> |"Hidden State<br>(h₀, c₀)"|C["GPT-2 Decoder"]
T--> P["GPT-2 Tokenizer"]
P -.-> C

C --> F["Next Token<br>p(yₜ|yₜ₋₁, h) = 'e'"] -.-> T
style T fill:#e9b116,stroke:none,color:black


```
---
# Evaluation
To properly evaluate the model, we decided to use the following evaluation techniques:

| Evaluation Technique | Explanation         |
|------------------------|---------------|                                                              
| **BLEU**         | BLEU (Bilingual Evaluation Understudy) is a metric used to evaluate how closely a machine-generated sentence matches a reference translation. In ASLens, it helps assess how accurately our AI translates ASL into written English by comparing word sequences. It is useful for checking the overall quality and fluency of the translation.         |
| **METEOR**       | METEOR evaluates translations based on word matches, synonyms, and word order, making it more flexible than BLEU. This is helpful because ASL does not always follow standard English grammar, so METEOR better captures translations that are semantically correct. It ensures the meaning is preserved even if the wording varies.|
| **ROUGE-1**      | ROUGE-1 measures the overlap of individual words between the model output and the reference. It shows whether the essential words from the correct translation are being included, helping confirm that the model captures key vocabulary from ASL.|   
| **ROUGE-2**   |ROUGE-2 looks at overlapping word pairs, providing insight into how well the model preserves short phrases. This matters in ASLens since phrase structure affects clarity and readability. It helps evaluate whether the output sounds natural and flows correctly.|     
| **ROUGE-L**   |ROUGE-L focuses on the longest matching sequence of words, reflecting how well the sentence structure is maintained. For ASLens, this helps determine if the overall order of translated signs makes sense in English. It is valuable for checking the fluency and coherence of the output.|
| **WER**   |WER (Word Error Rate) measures the number of errors—substitutions, deletions, and insertions—between the predicted and reference text, relative to the total number of words. In ASLense, it helps us quantify how many words the model gets wrong when translating ASL into written English. This metric is especially useful for tracking accuracy and identifying areas where the model consistently misinterprets or misses signs.|     

We tested each of our models on the test data and find evaluation metrics. We wi

## Experiment 1: Encoder-Decoder Architecture with CharRNN as Decoder
<div align ="center">

| Evaluation Technique | Value         | Expected Range         |
|------------------------|---------------|---------------|                                                         
| **BLEU**         | 0.01168| 	20–50 (higher is better)|
| **METEOR**       | 0.08298|0.4–0.7 (higher is better)	|
| **ROUGE-1**      | 0.12310| 0.4–0.6 (higher is better)	 | 
| **ROUGE-2**      | 0.01155|  0.2–0.4 (higher is better)	   |
| **ROUGE-L**      | 0.09490| 0.3–0.5 (higher is better)	|
| **WER**          | 0.78479| 20–50% (lower is better)	  |
</div>


## Experiment 2: Encoder-Decoder Architecture with GPT-2 as Decoder
<div align ="center">

| Evaluation Technique | Value         | Expected Range         |
|------------------------|---------------|---------------|                                                         
| **BLEU**         | 0.01389| 	20–50 (higher is better)|
| **METEOR**       | 0.07141|0.4–0.7 (higher is better)	|
| **ROUGE-1**      | 0.15781| 0.4–0.6 (higher is better)	 | 
| **ROUGE-2**      | 0.01245|  0.2–0.4 (higher is better)	   |
| **ROUGE-L**      | 0.11316| 0.3–0.5 (higher is better)	|
| **WER**          | 0.78125| 20–50% (lower is better)	  |
</div>

### Comparison of Metrics: Experiment 1 vs. Experiment 2

![](https://github.com/ASLense/ASLens/blob/main/assets/metrics.png)

## Human Evaluation
These metrics cannot be directly used to fully determine the model’s performance. Therefore, we conducted a self-evaluation, where we manually inspected several examples from the test set and analyzed the generated outputs. We observed that neither of the models perfectly matches the expected sentences (as also reflected in the metric values). Instead, both models tend to hallucinate outputs influenced by the general theme of the sign language video.

For example, if a video is about swimming, the model might generate text related to the sea, beach, or vacation—capturing the broader context but not the precise intended sentence. This kind of behavior is especially common with the model from **Experiment 1**.

On the other hand, the model from **Experiment 2** produces more fluent and meaningful sentences, often with clearer sentiment, though it too tends to hallucinate in a way that aligns with the general context of the input.
# Conclusion
We can conclude that the models have learned certain patterns and context from the videos. However, due to limitations in data and computational resources, the models are not yet able to consistently produce accurate and precise transcriptions. 

We will certainly continue working on this project and aim to improve our models by adding attention mechanisms to help the system learn more precise patterns and relationships in the input data. We are also considering using Reinforcement Learning from Human Feedback (RLHF), as we observed that our models tend to generate text that aligns with the general context rather than producing specific, accurate transcriptions.

# Poster
![](https://github.com/ASLense/ASLens/blob/main/assets/poster.png)
