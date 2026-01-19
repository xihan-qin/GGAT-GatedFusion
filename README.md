# Multichannel-Gated-Graph-Attention-Networks-for-Disease-Comorbidity

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


## Models and Usage
This project is designed to be run in Google Colab. All dependencies are handled
within the notebooks. To reproduce the experiments, open the notebooks in Google
Colab, mount your Google Drive, and update the project root path in the notebook
to your own Drive location. 

Specifically, in the *Mount Drive* block:
```python
os.chdir('/content/drive/My Drive/Colab_Notebooks/[your-project-root]')
```

### GGAT Single-Channel Models

The notebook `GGAT_singlechannel.ipynb` implements GGAT single-channel models, including:

- **GGAT-Connect** (`model_type = "n2v"`)
- **GGAT-Disease** (`model_type = "label"`)

#### Runtime Notes
- The default number of training epochs is **3000**, which requires a GPU and takes
  approximately **9 hours** to complete on Google Colab.
- For users who only want to **verify that the code runs correctly** or to
  **inspect the model design**, the number of epochs can be reduced (e.g., `epochs = 1`),
  in which case the notebook will finish within a few minutes.
- Full-length training is only required to **reproduce the reported performance**.

#### GGAT single-channel framework
![fig1](https://github.com/xihan-qin/GGAT-GatedFusion/blob/master/figs/GGAT%20channel.png)

---

### GGAT-GatedFusion Model

The notebook `GGAT_fusion.ipynb` implements the **GGAT-GatedFusion** model.

#### Important Notes
- This notebook requires **pre-trained representations** learned from the
  single-channel GGAT models.
- These representations are saved as `.pt` files during execution of
  `GGAT_singlechannel.ipynb`.

For convenience and reproducibility, the required `.pt` files are **provided in this
repository**, allowing users to directly run the fusion model **without re-training**
the single-channel models.

#### GatedFusion
![fig2](https://github.com/xihan-qin/GGAT-GatedFusion/blob/master/figs/gatedfusion.png)

## Results Showcase
### Performance of Disease Comorbidity Prediction: Baselines and GGAT models
Results are reported as mean ± standard deviation across cross-validation folds.
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

### Gated and Non-Gated Model Variants on the Disease Channel
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
      <td><b>0.8217 ± 0.0219</b></td>
      <td><b>0.8610 ± 0.0194</b></td>
      <td><b>0.7476 ± 0.0229</b></td>
      <td><b>0.4945 ± 0.0366</b></td>
    </tr>
    <tr>
      <td>GGCN</td>
      <td>0.8072 ± 0.0175</td>
      <td>0.8530 ± 0.0134</td>
      <td>0.7311 ± 0.0183</td>
      <td>0.4654 ± 0.0309</td>
    </tr>
    <tr>
      <td>GGraphSAGE</td>
      <td>0.8105 ± 0.0249</td>
      <td>0.8564 ± 0.0220</td>
      <td>0.7374 ± 0.0217</td>
      <td>0.4759 ± 0.0381</td>
    </tr>
    <tr>
      <td colspan="5" align="center"><b>Non-Gated Variants</b></td>
    </tr>
    <tr>
      <td>GAT</td>
      <td>0.5685 ± 0.0421</td>
      <td>0.5796 ± 0.1825</td>
      <td>0.5696 ± 0.0241</td>
      <td>0.1253 ± 0.0593</td>
    </tr>
    <tr>
      <td>GCN</td>
      <td>0.7715 ± 0.0172</td>
      <td>0.8185 ± 0.0168</td>
      <td>0.7040 ± 0.0158</td>
      <td>0.4100 ± 0.0247</td>
    </tr>
    <tr>
      <td>GraphSAGE</td>
      <td>0.8169 ± 0.0224</td>
      <td>0.8601 ± 0.0216</td>
      <td>0.7390 ± 0.0158</td>
      <td>0.4817 ± 0.0396</td>
    </tr>
  </tbody>
</table>

### UMAP projections of learned features at different stages of the GGAT framework. 
(a,b) Protein-level features from the connectivity and disease channels after GGAT layers. (c,d) Disease–pair features after adaptive pooling, where informative nodes are emphasized to form more structured representations. (e) Fused disease-pair features from the gated fusion model, integrating complementary signals from the two channels. Dashed/dotted outlines highlight clusters of pairs that involve selected diseases (glioma, coronary disease, and macular degeneration), and the annotated points indicate representative disease pairs used for downstream network inspection. (f,g) Two-hop interactome neighborhoods for representative disease pairs. Nodes correspond to proteins associated with each disease and their shared/intermediate neighbors, and edges represent protein–protein interactions.
![fig3](https://github.com/xihan-qin/GGAT-GatedFusion/blob/master/figs/fusion_umap_grid%20.png)

### Network visualization of disease pairs involving coronary disease (CAD)
![fig4](https://github.com/xihan-qin/GGAT-GatedFusion/blob/master/figs/CAD_TP_vs_TN_side_by_side.png)

### Note
For additional details, please consult the paper titled "Gated Graph Attention with Multichannel Fusion for Interactome-Based Disease Comorbidity Prediction"