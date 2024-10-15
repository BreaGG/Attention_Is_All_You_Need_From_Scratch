# **Attention Is All You Need: Transformer from Scratch**

## **Project Aim**

The goal of this project is to have a deep understanding of deep learning concepts implementing a Transformer model from scratch using PyTorch. The Transformer, introduced in the groundbreaking paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), revolutionized sequence modeling, especially in natural language processing (NLP) tasks like machine translation. This project walks through the steps to implement the Transformer architecture, understand its mathematical foundations, and apply it to a practical task—translating between languages using the **Multi30k** English-German dataset.

## **Table of Contents**

1. [Project Aim](#project-aim)
2. [Mathematical Foundations](#mathematical-foundations)
    - Attention Mechanism
    - Scaled Dot-Product Attention
    - Multi-Head Attention
    - Positional Encoding
    - Encoder and Decoder
3. [Model Components](#model-components)
    - Embedding Layer
    - Multi-Head Attention
    - Feedforward Network
    - Encoder
    - Decoder
    - Transformer
4. [Setup and Installation](#setup-and-installation)
5. [Training Procedure](#training-procedure)
6. [Dataset](#dataset)
7. [Usage](#usage)
8. [License](#license)

## **Mathematical Foundations**

### **1. Attention Mechanism**
The attention mechanism allows the model to focus on specific parts of the input sequence when generating output, which helps when dealing with long sequences. Each token in the input sequence is assigned a score representing its importance relative to other tokens.

The attention mechanism is computed as follows:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:
- \( Q \) (Query): The matrix representing the token currently being attended to.
- \( K \) (Key): The matrix representing all the tokens in the sequence.
- \( V \) (Value): The matrix containing the values corresponding to each token.
- \( d_k \): The dimensionality of the keys, used to scale the dot product.

### **2. Scaled Dot-Product Attention**
The Scaled Dot-Product Attention takes the query and key matrices and computes their dot product. It is then scaled by \( \frac{1}{\sqrt{d_k}} \) to stabilize gradients during backpropagation.

### **3. Multi-Head Attention**
Multi-Head Attention allows the model to focus on multiple parts of the sequence simultaneously. Instead of performing a single attention operation, the input is split into multiple "heads," and attention is applied to each. The outputs are concatenated back together:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W^O
$$

Each head is a separate instance of the Scaled Dot-Product Attention mechanism.

### **4. Positional Encoding**
Since Transformers do not have any recurrence or convolution, positional encoding is used to provide the model with information about the order of tokens in a sequence. This encoding is added to the token embeddings.

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$
$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

Where:
- \( pos \) is the position of the token in the sequence.
- \( i \) is the index of the dimension.

### **5. Encoder and Decoder**
The Transformer uses an encoder-decoder architecture. Each encoder layer contains:
- A multi-head self-attention mechanism.
- A feedforward network.

The decoder layers are similar but include an additional layer to attend to the encoder output.

## **Model Components**

### **1. Embedding Layer**
This layer converts input tokens into dense vectors of fixed size.

### **2. Multi-Head Attention**
The attention mechanism is split into several heads. The Scaled Dot-Product Attention is applied to each head, and the results are concatenated.

### **3. Feedforward Network**
A simple fully-connected neural network applied after each attention block to increase the model's capacity.

### **4. Encoder**
The encoder is a stack of multiple layers. Each layer contains a multi-head attention mechanism and a feedforward network.

### **5. Decoder**
The decoder is also a stack of multiple layers. In addition to the self-attention, it has an extra attention mechanism to attend to the encoder output.

### **6. Transformer Model**
Combines the encoder and decoder to form the complete model, where the encoder processes the input sequence, and the decoder generates the output sequence.

## **Setup and Installation**

To run this project, you'll need to set up a Python environment and install the necessary dependencies.

### **1. Clone the Repository**
```bash
git clone https://github.com/BreaGG/Attention_Is_All_You_Need_From_Scratch
cd Attention_Is_All_You_Need_From_Scratch
```

### **2. Set Up a Virtual Environment (optional)**
You can use Python's `venv` or `conda` to create a virtual environment for the project.

```bash
python3.10 -m venv transformer-env
source transformer-env/bin/activate  # On Windows: .\transformer-env\Scripts\activate
```

### **3. Install Dependencies**
Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

If you don't have the language models, install them using `spaCy`:

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### **4. Directory Structure**
The key files are:
- `data_preparation.py`: Handles loading and tokenizing the dataset.
- `train_transformer.py`: Defines the training loop and the Transformer model.
- `transformer_model.py`: Contains the model implementation.
- `requirements.txt`: Defines the dependencies for the project.

## **Training Procedure**

### **1. Data Preparation**
The script `data_preparation.py` prepares the data using the Multi30k dataset for English-to-German translation.

```bash
python data_preparation.py
```

This script tokenizes the data, builds the vocabulary, and creates data loaders for the training process.

### **2. Training the Model**
The script `train_transformer.py` handles the training process:

```bash
python train_transformer.py
```

The model will be trained using the cross-entropy loss function, with padding tokens ignored. The training process uses backpropagation through time (BPTT), and the optimizer used is Adam.

## **Dataset**

The **Multi30k** dataset is used for training and evaluation. It consists of English-German sentence pairs.

The dataset is downloaded and preprocessed using `torchtext`. The tokenization is handled by `spaCy`.

## **Usage**

1. **Run Data Preparation**:
   ```bash
   python data_preparation.py
   ```

2. **Train the Model**:
   ```bash
   python train_transformer.py
   ```

3. **Adjust Hyperparameters**:
   Modify the `train_transformer.py` script to adjust the number of epochs, learning rate, batch size, etc.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
