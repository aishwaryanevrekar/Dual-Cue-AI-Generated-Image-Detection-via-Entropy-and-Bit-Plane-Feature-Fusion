# Pure LOTA (ResNet-50) Rebuild Results

I completely tore down the old LOTA architecture and rebuilt it strictly according to the **Wang et al. (ICCV 2025)** paper specification.

## Architectural Changes Made
1. **Single Patch Extraction**: I replaced the custom quadrant diversity logic with Equation 6 from the paper. The network now extracts exactly **one 32x32 patch** that contains the absolute highest gradient divergence (the noisiest patch).
2. **Nearest-Neighbor Upscaling**: The 32x32 binary noise patch is mathematically upscaled to 256x256 before being sent to the classifier, perfectly preserving the 0/255 blocky noise structure of the LSBs.
3. **ResNet-50 Backbone**: I removed the custom 400K-parameter shallow CNN and injected a massive **23 Million parameter ResNet-50** pre-trained on ImageNet.

## Training Progress

Because the model jumped from 400,000 parameters to 23,500,000 parameters, and we are processing 256x256 upscaled patches, the training is heavily utilizing your NVIDIA GTX 1050 Ti.

The training is currently running in the background. Each epoch takes approximately **4 minutes** to compute (after the initial ResNet-50 weights download which took 6 mins). 

**Epoch 1 Results:**
* **Train Accuracy**: `50.72%`
* **Val Accuracy**: `49.35%`

**Epoch 2 Results:**
* **Train Accuracy**: `50.08%`
* **Val Accuracy**: `50.05%`

**Epoch 3 Results:**
* **Train Accuracy**: `51.12%`
* **Val Accuracy**: `49.65%`

**Epoch 4 Results:**
* **Train Accuracy**: `50.57%`
* **Val Accuracy**: `51.05%`

**Epoch 5 Results:**
* **Train Loss**: `0.6901` | **Acc**: `54.08%`
* **Val Loss**: `0.6926` | **Acc**: `52.30%`

### Final Test Set Evaluation
* **Test Accuracy**: `52.50%`
* **Test ROC-AUC**: `0.5304`
* **Test Precision**: `0.5231`
* **Test F1-Score**: `0.5437`

> [!WARNING]
> **Issue Identified!** The accuracy is abnormally low (52.5% is near random guessing). I have identified the root cause: The ResNet-50 backbone is pre-trained on ImageNet, which expects normalized input tensors in the `[0.0, 1.0]` range. However, our LOTA extractor is outputting raw `[0.0, 255.0]` binary activations. Passing `255.0` directly into an ImageNet backbone completely blows up the internal BatchNormalization statistics and gradients, preventing the model from learning!
