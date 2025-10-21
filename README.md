# 🌱 Plant Spatio-Temporal Integration Network (PSTN)
### Unified Framework for Temporal Spatial Transcriptomic Integration  
**Version:** Training Script (with Evaluation Alignment)  
**Author:** [Yuheng Zhu]  
**Affiliation:** Jilin University  
**Date:** 2025  

---

## 🧠 English Summary

**PSTN** (Plant Spatio-Temporal Integration Network) is a unified model designed to jointly optimize mapping matrices across multiple time points (e.g., 0h, 12h, 24h).  
It integrates single-cell and spatial transcriptomic data by enforcing **temporal consistency** during training, enabling smooth tissue trajectory reconstruction across infection stages.

This version is fully compatible with the evaluation scripts, ensuring consistent preprocessing, normalization, and loss computation.

---

### 🚀 Key Features
- 🔗 **Joint Optimization:** Learn cell-to-space mappings `{M_t}` for all time points simultaneously.  
- ⏱️ **Temporal Regularization:** Apply bidirectional cosine similarity between adjacent time points.  
- 🧬 **Biological Context:** Designed for plant–pathogen interaction datasets (e.g., *Magnaporthe oryzae* infection in rice).  
- 📊 **Evaluation Alignment:** Output files (`A_raw.npy`, `B.npy`, `mapping_matrix.pt`) match evaluation scripts exactly.  

---

### 📂 File Structure
