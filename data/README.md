## download codebert-base model

```bash
# Download the pre-trained CodeBERT model
mkdir codebert-base && cd codebert-base
wget https://huggingface.co/microsoft/codebert-base/resolve/main/pytorch_model.bin
wget https://huggingface.co/microsoft/codebert-base/resolve/main/config.json
wget https://huggingface.co/microsoft/codebert-base/resolve/main/merges.txt
wget https://huggingface.co/microsoft/codebert-base/resolve/main/tokenizer_config.json
wget https://huggingface.co/microsoft/codebert-base/resolve/main/vocab.json
```

## datasets

You can get MSR_data_cleaned.zip from https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing(in https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset)

metric.csv file is a mapping from cve id to labels,which is gotten by crawling the https://nvd.nist.gov/