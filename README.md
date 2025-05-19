# Recognizing Underlying Patterns in Categorical Data via Symbolization and Masking Mechanisms

```powershell
- Depending on your transformer toolkit versions, the transformer import code may need to be adjusted, like as follows:
+ from transformers.modeling_bert import BertPreTrainedModel, BertPooler
+ --> from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertPooler
- (Please check your transformer toolikt, and update the import code accordingly.)
```

## How to run the code?

After downloading the code, you can run

```powershell
python3 run.py
```

directly for categorical clustering. We suggest adjusting the hyperparameters multiple times to achieve better results.

## What are the scripts used for?

(1)[LM/BertForMaskedLM](https://github.com/kcisgroup/SAMM/tree/main/LM/BertForMaskedLM): Contains the model structure and configuration of the BERT.

(2)[make_dataset](https://github.com/kcisgroup/SAMM/tree/main/make_dataset): Data processing. Help us prepare the training set.

(3)[models](https://github.com/kcisgroup/SAMM/tree/main/model): Define the network structure of SAMM.

(4) [utils](https://github.com/kcisgroup/SAMM/tree/main/utils): Contains functions for data processing and model evaluation.

## Several toolkits may be needed to run the code

(1) pytorch (https://anaconda.org/pytorch/pytorch)

(2) sklearn (https://anaconda.org/anaconda/scikit-learn)

(3) transformers (https://anaconda.org/conda-forge/transformers)