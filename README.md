# ✨ Transformer Decoder from Scratch – Tiny Shakespeare

> **“I built a GPT‑like decoder architecture from the ground up—no shortcuts, no pre‑made models. Just raw PyTorch, math, and curiosity.”**

This project is a hands‑on deep dive into transformer decoders. I wanted to understand every internal—multi‑head attention, feed‑forward nets, layer norms, training dynamics, and visualization. So I built it from scratch and trained it on the Tiny Shakespeare dataset to generate text.  
This is not just a demo. It’s a **reproducible, educational, and extensible framework** for decoder‑only models.

---

## 📌 Highlights

- ✅ **From‑scratch decoder architecture** (Multi‑Head Attention, FFN, LayerNorm, Dropout, Masking)  
- 📚 **Trained on Tiny Shakespeare** with a real tokenizer (`bert‑base‑uncased`)  
- 🎯 Implements **causal attention**, weight initialization, positional embeddings, and a manual training loop  
- 🎨 **Attention visualization** tool to inspect how the model distributes focus across tokens  

---

## 🔍 What’s Inside

| Module                                   | Description                                               |
|------------------------------------------|-----------------------------------------------------------|
| Import_libraries.py                      | All necessary imports (PyTorch, Transformers, etc.)       |
| Feed_Forward_Network.py                  | Two‑layer FFN with Xavier init and ReLU                   |
| MultiHeadAttn.py                         | Per‑head multi‑head attention with masking                |
| Decoder_Block.py                         | Decoder block combining MHA + FFN + LayerNorm + residuals |
| FC_Layer(Final_layer).py                 | Final linear projection to vocabulary size                |
| DecoderOnlyModel(...).py                 | Assembles full decoder: token + positional embeddings     |
| Train.py                                 | Custom training loop (tokenization → dataloader → train)  |
| atten_image.py                           | Attention heatmap visualizer per layer & head             |
| tiny_transformer_weights.pth             | Trained model weights                                     |
| Requirement.txt                          | Python dependencies                                       |

---

## 🏗️ Model Architecture

> **I didn’t use** nn.Transformer **or** nn.MultiheadAttention—everything is built manually.

Each DecoderBlock consists of:

1. **Multi‑Head Attention** (from scratch) – Q/K/V per head, scaled dot‑product, causal mask  
2. **Residual Connections** & **LayerNorm**  
3. **Feed‑Forward Network** (two linear layers + ReLU)  
4. **Dropout** between sub‑layers  
5. **Final linear projection** to vocabulary size  

---

## 📈 Training

The model is trained to predict the next token in the Tiny Shakespeare corpus.

# Run training
python Train.py

Dataset: Tiny Shakespeare (via HuggingFace)
Loss: CrossEntropy
Optimizer: AdamW (lr=3e‑4)
Epochs: 25
Batch Size: 16
Sequence Length: 64
Model Dim (d_model): 128
Heads: 4
After training, weights are saved to:
tiny_transformer_weights.pth


## 🎨 Attention Visualization
Inspect how each attention head focuses across tokens:
-atten_image

![Transformer Decoder Architecture](https://raw.githubusercontent.com/Umang-projects/-Transformer-Decoder-from-Scratch-Tiny-Shakespeare-/main/atten_image.png)


## 🤔 Why I Built This
-I didn’t want to just use transformers—I wanted to understand them end‑to‑end:
-Reinforce the theory behind attention mechanisms.
-Practice raw PyTorch modeling without high‑level abstractions.
-Create reusable building blocks for GPT‑style models.
