
# ML Assignment

**Self-Supervised Image Fragment Matching via Contrastive Representation Learning**

This project trains a neural encoder to produce meaningful embeddings for image **fragments**, such that fragments originating from the same image cluster together in embedding space. It uses self-supervised contrastive learning or pairwise binary cross-entropy (BCE) loss, and evaluates performance using metrics such as ARI, AUC, and MCC.

---

## 🧪 Evaluation

### ➤ Run metrics on the validation set

By default, this samples **10 images at a time**, repeating the process 50× to robustly evaluate the model on ~500 images (10% of the full validation set):

```bash
python src/collect_metrics.py --data_path "data" --model_checkpoint "model/best_model_CL_8.keras"
````

This evaluates the model by:

* Reconstructing full images from fragments using **KMeans clustering** and pairwise distances from embeddings
* Measuring **Adjusted Rand Index (ARI)** to quantify clustering accuracy
* Reporting **MCC** (Matthews Correlation Coefficient) and **AUC** (ROC) using fragment similarity scores

---

### ➤ Visualize fragment embeddings

To visualize clustering and similarities for a **single batch of 10 validation images**, run:

```bash
python src/visualize_only.py --data_path "data" --model_checkpoint "model/best_model_CL_8.keras"
```

This will generate:

* Two histograms showing the distance distribution between fragments from the same image vs. across different images
* A 2D projection (UMAP or PCA) of fragment embeddings colored by image ID
* Two similarity matrix (With indices ordered by images or by clusters)
* Plots saved to the `Figure/` directory

---

## 🔁 Training

To retrain the model from scratch using either contrastive loss (NT-Xent) or pairwise BCE:

```bash
# For contrastive loss (NT-Xent)
python src/main.py --loss CL

# For BCE loss
python src/main.py --loss BCE
```

Checkpoints will be saved to the `model/` folder as:

* `best_model_CL_<dim>.keras`
* `best_model_BCE_<dim>.keras`

You can change the embedding dimension or training steps by editing `train.py`.

---

## 📂 Project Structure

```
├── data/                  # ImageNet64 dataset - only validation data provided in this repo
├── model/                 # Saved model checkpoints
├── Figure/                # Evaluation plots
├── src/
│   ├── train.py           # Training and evaluation logic
│   ├── collect_metrics.py # Evaluation script
│   ├── visualize_only.py  # Visualization script
├── README.md
├── requirements.txt
```

---

## 📋 Requirements

* Python 3.8+
* TensorFlow 2.x (with GPU support recommended)
* scikit-learn
* matplotlib
* umap-learn
* tqdm

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🧠 Method Overview

* Each 64×64 image is split into 4×4 = 16 fragments (16×16 each)
* The encoder maps each fragment to a fixed-length embedding
* Training encourages fragments from the same image to be closer in embedding space

Two training losses available:

* `nt_xent_loss`: Normalized Temperature-scaled Cross Entropy (contrastive loss)
* `pairwise_bce_loss`: Binary classification on all fragment pairs (positive/negative)


