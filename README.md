# 🧗‍♂️ Kilter Board AI: Deep Learning Route Analyzer

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A machine learning pipeline for analyzing, reconstructing, and predicting the difficulty (V-Grade) of indoor climbing routes on the Kilter Board. The project processes raw SQLite data into multi-channel image tensors and leverages a Convolutional Autoencoder combined with an MLP regression head to evaluate route difficulty based on geometric hold layouts.

---

## ⚙️ Architecture & Pipeline

### 1. Data Processing
* Extracts raw route strings (e.g., `p14,p15`) from the official Kilter SQLite database.
* Converts textual hold data into `[4, 173, 185]` PyTorch tensors.
* Channels are separated by hold type: Start, Middle, Finish, and Foot holds.

### 2. Convolutional Autoencoder (Feature Extraction)
* **Encoder:** A series of `Conv2d` layers reducing the `173x185` board matrix into a condensed 128-dimensional latent representation.
* **Decoder:** Upsampling layers reconstructing the visual representation of the route.
* Trained on tens of thousands of routes to capture the spatial geometry and hold density of the Kilter Board.

### 3. Grade Predictor (Regression Head)
* Utilizes the frozen weights of the pre-trained Encoder.
* Replaces the Decoder with a fully connected Multi-Layer Perceptron (`128 -> 64 -> 16 -> 1`).
* Trained using Mean Squared Error (MSE) to map the 128-d geometric feature vector to community-voted difficulty grades.

---

## 📊 Performance Metrics

Evaluated on a test set of 10,000 routes (40-degree wall angle):
* **Mean Absolute Error (MAE):** `2.47` (Approx. 1.5 V-Grades)
* **Predicted within 1 V-Grade (< 1.6 pts error):** `38.5%`
* **Exact Matches (< 0.5 pts error):** `12.3%`

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* PyTorch
* Matplotlib, NumPy

### Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/KilterClimbAI.git
   cd KilterClimbAI
### Usage
Run the visualization script to randomly sample a route, reconstruct it, and output the model's grade prediction:
```bash
  python src/scripts/visualize.py
```
### 4. Download Pre-trained Models
The repository does not contain the heavy `.pth` weight files. To run the predictions immediately:
1. Go to the [Releases](../../releases) tab of this repository.
2. Download `kilter_ae_weights.pth` and `kilter_predictor_weights.pth`.
3. Place both files inside the `models/` directory.

### 🔮 Future Scope
- Route Generation: Sampling the 128-d latent space to procedurally generate novel boulder problems.
- Interactive UI: A web-based interface for real-time grade evaluation of custom-built routes.

