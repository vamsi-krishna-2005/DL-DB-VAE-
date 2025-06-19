# 🎯 Debiased Face Classification using DB-VAE

This project explores how Variational Autoencoders (VAEs) can help mitigate dataset bias in facial classification. Inspired by real-world fairness concerns, we design a **Debiasing Variational Autoencoder (DB-VAE)** that adaptively samples underrepresented face examples.

---

## 📦 Dataset

- **Positive (faces)**: Subset of CelebA
- **Negative (non-faces)**: Subset of ImageNet / CIFAR-10
- **Bias labels**: Face brightness (Lighter / Darker skin tones)

---

## ⚙️ Working Principle

### ✅ 1. **Standard CNN Classifier**

- Trains on positive + negative examples
- Learns to classify face vs. not face
- May be biased toward over-represented groups (e.g. lighter skin tones)

### ✅ 2. **Variational Autoencoder (VAE)**

- Trained only on face images
- Learns a **latent representation** (features) of faces in an unsupervised way
- Outputs: `μ` (mean vector), `σ` (std dev vector), reconstructed image

### ✅ 3. **PCA + Latent Space Visualization**

- PCA used to project latent vectors to 2D
- Samples colored by brightness → reveals clusters of light/dark faces
- Shows imbalance: some areas are sparse (under-represented)

### ✅ 4. **Adaptive Sampling Strategy**

- Density of samples in latent space is calculated
- Sampling probability for training set ∝ `1 / (density + ε)`
- **Rare faces** (like dark-skinned ones) get sampled **more often**

### ✅ 5. **DB-VAE Classifier**

- CNN is retrained using batches drawn adaptively based on rarity
- Performance improves on rare groups
- Bias is reduced, fairness improved without labels

---

## 🔬 Results

| Model             | Validation Accuracy | Notes                                     |
| ----------------- | ------------------- | ----------------------------------------- |
| Standard CNN      | 53%                 | Biased toward lighter faces               |
| DB-VAE Classifier | 99%                 | Fairer + performs better on rare features |

- ✅ Adaptive sampling visualized in PCA
- ✅ DB-VAE improves generalization across skin tone subgroups

---

## 🛠️ Tools Used

- TensorFlow / Keras
- NumPy, matplotlib, seaborn
- PCA (scikit-learn)
- CelebA dataset

## 📚 Learnings

- VAEs can uncover latent structure in unlabeled data
- Sampling based on latent rarity can correct dataset imbalance
- DB-VAE shows improved fairness without needing skin tone labels

---![DB-VAE Overview](https://github.com/user-attachments/assets/351a347c-5faa-40fc-9819-dc6efef04419)

