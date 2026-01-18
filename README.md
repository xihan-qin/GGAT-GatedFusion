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

```markdown
Specifically, in the *Mount Drive* block, modify the working directory, e.g.,
python
os.chdir('/content/drive/My Drive/Colab_Notebooks/[your-project-root]')
```

* `GGAT_singlechannel.ipynb`: GGAT single-channel models, including GGAT-Connect
  (`model_type = "n2v"`) and GGAT-Disease (`model_type = "label"`).

* `GGAT_fusion.ipynb`: GGAT-GatedFusion model with multichannel gated fusion.