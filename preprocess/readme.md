## Data preparation
- All datasets are publically available and can be accessed at [WOS](https://github.com/kk7nc/HDLTex), [RCV1-V2](https://trec.nist.gov/data/reuters/reuters.html).
- We followed the specific details mentioned in the  [contrastive-htc](https://github.com/wzh9969/contrastive-htc#preprocess) repository to preprocess the original datasets (WOS and RCV1-V2). 
### 1. WOS
- The original raw dataset is added to the WOS folder. Run the following scripts to obtain the preprocessed and tokenized version of the dataset
    ```bash
    cd preprocess/WOS
    python preprocess_wos.py
    python data_wos.py
### 2. RCV1-V2
- The original raw dataset is available after signing an [agreement](https://trec.nist.gov/data/reuters/reuters.html). Place the downloaded `rcv1.tarz.xz` file in the `preprocess\RCV` folder and run the following scripts to obtain the preprocessed and tokenized version of the dataset
    ```bash
    cd preprocess/RCV
    python preprocess_rcv1.py
    python data_rcv1.py
