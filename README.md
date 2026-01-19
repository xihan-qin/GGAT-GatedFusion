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
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>AUROC</th>
      <th>AUPRC</th>
      <th>Accuracy</th>
      <th>MCC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="5" align="center"><b>Gated Variants</b></td>
    </tr>
    <tr>
      <td><b>GGAT</b></td>
      <td><b>0.8217 ± 0.0189</b></td>
      <td><b>0.8595 ± 0.0186</b></td>
      <td><b>0.7485 ± 0.0201</b></td>
      <td><b>0.4979 ± 0.0337</b></td>
    </tr>
    <tr>
      <td>GGCN</td>
      <td>0.8176 ± 0.0162</td>
      <td>0.8548 ± 0.0146</td>
      <td>0.7469 ± 0.0156</td>
      <td>0.4885 ± 0.0301</td>
    </tr>
    <tr>
      <td>GGraphSAGE</td>
      <td>0.8085 ± 0.0231</td>
      <td>0.8524 ± 0.0238</td>
      <td>0.7374 ± 0.0247</td>
      <td>0.4750 ± 0.0443</td>
    </tr>
    <tr>
      <td colspan="5" align="center"><b>Non-Gated Variants</b></td>
    </tr>
    <tr>
      <td>GAT</td>
      <td>0.5504 ± 0.0387</td>
      <td>0.6226 ± 0.0261</td>
      <td>0.5685 ± 0.0461</td>
      <td>0.1115 ± 0.0635</td>
    </tr>
    <tr>
      <td>GCN</td>
      <td>0.7828 ± 0.0202</td>
      <td>0.8378 ± 0.0200</td>
      <td>0.7138 ± 0.0190</td>
      <td>0.4329 ± 0.0300</td>
    </tr>
    <tr>
      <td>GraphSAGE</td>
      <td>0.8061 ± 0.0255</td>
      <td>0.8553 ± 0.0233</td>
      <td>0.7280 ± 0.0199</td>
      <td>0.4617 ± 0.0445</td>
    </tr>
  </tbody>
</table>

