# Gated-Graph-Attention-for-Disease-Comorbidity

In this work, we propose a Gated Graph Attention Network (GGAT) framework for
interactome-based disease comorbidity prediction. GGAT performs topology-aware
local attention over protein–protein interaction networks and incorporates
gating mechanisms to regulate information flow and stabilize representation
learning on noisy biological graphs. In addition, a multichannel fusion strategy
integrates protein connectivity and disease association signals to capture
complementary biological information. Extensive experiments on benchmark
datasets demonstrate that GGAT consistently outperforms state-of-the-art
Transformer-based baselines across multiple evaluation metrics, highlighting
its effectiveness and flexibility for modeling disease–disease relationships.


## Usage

This project is designed to be run in Google Colab. All dependencies are handled
within the notebooks. To reproduce the experiments, open the notebooks in Google
Colab, mount your Google Drive, and update the project root path in the notebook
to your own Drive location. 

Specifically, in the *Mount Drive* block:
```python
os.chdir('/content/drive/My Drive/Colab_Notebooks/[your-project-root]')
```

* `GGAT_singlechannel.ipynb`: GGAT single-channel models including
  * GGAT-Connect (`model_type = "n2v"`)
  * GGAT-Disease (`model_type = "label"`)

* `GGAT_fusion.ipynb`: GGAT-GatedFusion model 


## GGAT single-channel framework
![fig1](https://github.com/xihan-qin/GGAT-GatedFusion/blob/master/figs/GGAT%20channel.png)

## GatedFusion
![fig2](https://github.com/xihan-qin/GGAT-GatedFusion/blob/master/figs/gatedfusion.png)

## Results Showcase
### Performance of Disease Comorbidity Prediction: Baselines and GGAT models
| Model              | AUROC               | AUPRC               | Accuracy            | MCC                |
|--------------------|---------------------|---------------------|---------------------|--------------------|
| GE                 | 0.5497 ± 0.0079     | –                   | 0.6150 ± 0.0078     | –                  |
| BSE-SVM            | 0.6469 ± 0.0183     | –                   | 0.6801 ± 0.0166     | –                  |
| TSPE-NoPE          | 0.7971 ± 0.0146     | 0.8429 ± 0.0168     | 0.7214 ± 0.0202     | 0.4340 ± 0.0299    |
| TSPE               | 0.8009 ± 0.0152     | 0.8438 ± 0.0199     | 0.7294 ± 0.0138     | 0.4578 ± 0.0378    |
| GGAT-Connect       | 0.8217 ± 0.0189     | 0.8595 ± 0.0186     | 0.7485 ± 0.0201     | 0.4979 ± 0.0337    |
| GGAT-Disease       | 0.8217 ± 0.0220     | 0.8610 ± 0.0194     | 0.7476 ± 0.0230     | 0.4945 ± 0.0220    |
| GGAT-EmbedFusion   | 0.8223 ± 0.0185     | 0.8599 ± 0.0165     | 0.7500 ± 0.0150     | 0.4975 ± 0.0333    |
| **GGAT-GatedFusion** | **0.8397 ± 0.0180** | **0.8758 ± 0.0175** | **0.7669 ± 0.0140** | **0.5337 ± 0.0310** |

Results are reported as mean ± standard deviation across cross-validation folds.

### Gated and Non-Gated Model Variants on the Connect Channel

| Model         | AUROC               | AUPRC               | Accuracy            | MCC                |
|---------------|---------------------|---------------------|---------------------|--------------------|
| **Gated Variants** |                     |                     |                     |                    |
| **GGAT**      | **0.8217 ± 0.0189** | **0.8595 ± 0.0186** | **0.7485 ± 0.0201** | **0.4979 ± 0.0337** |
| GGCN          | 0.8176 ± 0.0162     | 0.8548 ± 0.0146     | 0.7469 ± 0.0156     | 0.4885 ± 0.0301    |
| GGraphSAGE    | 0.8085 ± 0.0231     | 0.8524 ± 0.0238     | 0.7374 ± 0.0247     | 0.4750 ± 0.0443    |
| **Non-Gated Variants** |              |                     |                     |                    |
| GAT           | 0.5504 ± 0.0387     | 0.6226 ± 0.0261     | 0.5685 ± 0.0461     | 0.1115 ± 0.0635    |
| GCN           | 0.7828 ± 0.0202     | 0.8378 ± 0.0200     | 0.7138 ± 0.0190     | 0.4329 ± 0.0300    |
| GraphSAGE     | 0.8061 ± 0.0255     | 0.8553 ± 0.0233     | 0.7280 ± 0.0199     | 0.4617 ± 0.0445    |
