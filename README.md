# **HUNet: Homotopy Unfolding Network for Image Compressive Sensing**

HUNet is a deep unfolding network designed for **image compressive sensing (CS) reconstruction**.
It unfolds the iterations of a homotopy optimization algorithm into a multi-stage neural network, enhanced with multi-scale feature modeling and cross-stage feature fusion.
The method is published in **CVPR 2025**.

---

## 📌 **Overview**

Compressive sensing aims to recover high-quality images from a small number of measurements.
Traditional optimization-based CS methods are slow, and many deep learning approaches lack interpretability.
Deep unfolding networks (DUNs) bridge this gap by transforming iterative algorithms into trainable modules.

However, most existing DUNs only operate at a single spatial scale and often lose high-dimensional information across iterations.
HUNet addresses these limitations by introducing:

* **Homotopy unfolding framework**
* **Multi-scale feature learning**
* **Dual-path feature fusion across stages**

This leads to better convergence behavior and superior reconstruction quality.

---

## 🧠 **Method Summary**

HUNet consists of two major components:

### **1. Sampling Module**

* The input image is divided into blocks and sampled via a convolutional measurement operator.
* The sampling operator produces the CS measurement vector ( y ).

### **2. Reconstruction Module**

Reconstruction starts from an initial estimate ( x_0 ) using the transpose sampling operator and proceeds through **multiple unfolding stages**.

Each stage contains a **Multi-scale Homotopy Iterative Module (MHIM)** that performs:

* Multi-scale feature extraction
* Transformer-based local window modeling
* Homotopy continuation updates

Finally, a **Dual-path Feature Fusion Module (DFFM)** aggregates features across all stages to form the final reconstruction.

The overall architecture:
**Sampling → N × MHIM → DFFM → Output Image**

---

## 🌟 **Key Contributions**

* **Homotopy-based unfolding design**
  Merges classical homotopy continuation with modern deep learning for interpretability and strong performance.

* **Multi-scale feature extraction**
  Allows the model to capture local textures, edges, and global structures at different resolutions.

* **Dual-path feature fusion**
  Prevents feature degradation and ensures effective cross-stage information flow.

* **State-of-the-art performance**
  Outperforms existing deep models and unfolding methods on standard CS benchmarks.

---

## 📁 **Project Structure (example)**

> *Adjust this section if your repository structure differs.*

```
/models
    MHIM.py          # Multi-scale Homotopy Iterative Module
    DFFM.py          # Dual-path Feature Fusion Module
    HUNet.py         # Main model implementation
/dataset
    prepare_data.py  # Dataset preprocessing
/checkpoints
    hunet.pth        # Pretrained model (if provided)

test.py              # Evaluation / inference script
train.py             # Training script (if available)
utils.py             # Helper functions
```

---

## 🚀 **Usage**

### **Install Required Packages**

```
pip install -r requirements.txt
```

### **Run Inference**

Example:

```
python test.py \
    --model checkpoints/hunet.pth \
    --input data/measurements \
    --output results/
```

### **Training (if supported)**

```
python train.py --config configs/hunet.yaml
```

---

## 📄 **Citation**

If you use HUNet in your research, please cite:

```
@inproceedings{shen2025hunet,
  title={HUNet: Homotopy Unfolding Network for Image Compressive Sensing},
  author={Shen, F. and Gan, H.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

---

## 📘 **Reference**

Paper (CVPR 2025):
[https://openaccess.thecvf.com/content/CVPR2025/html/Shen_HUNet_Homotopy_Unfolding_Network_for_Image_Compressive_Sensing_CVPR_2025_paper.html](https://openaccess.thecvf.com/content/CVPR2025/html/Shen_HUNet_Homotopy_Unfolding_Network_for_Image_Compressive_Sensing_CVPR_2025_paper.html)

---

## 📝 **License**

Include the appropriate license here (MIT, Apache-2.0, etc.)


