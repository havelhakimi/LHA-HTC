# LHA-HTC
 Implementation for the 2023 IEEE International Conference on Big Data (BigData) accepted paper "**Label hierarchy alignment for improved hierarchical text classification**" [paper-link](https://ieeexplore.ieee.org/abstract/document/10386495)

## Requirements
- Python >= 3.7
- torch >= 1.6.0
- transformers >= 4.30.2
- Below libraries only if you want to run on GAT/GCN as the graph encoder
  - torch-geometric == 2.4.0
  - torch-sparse == 0.6.17
  - torch-scatter == 2.1.1

## Data
- All datasets are publically available and can be accessed at [WOS](https://github.com/kk7nc/HDLTex), [RCV1-V2](https://trec.nist.gov/data/reuters/reuters.html) and [NYT](https://catalog.ldc.upenn.edu/LDC2008T19). 
- We followed the specific details mentioned in the  [contrastive-htc](https://github.com/wzh9969/contrastive-htc#preprocess) repository to obtain and preprocess the original datasets (WOS and RCV1-V2).
- After accessing the dataset, run the scripts in the folder `preprocess` for each dataset separately to obtain tokenized version of dataset and the related files. These will be added in the `data/x` folder where x is the name of dataset with possible choices as: wos and rcv.
- For reference we have added tokenized versions of the WOS dataset along with its related files in the `data/wos` folder. Similarly do for rcv dataset.
- Detailed steps regarding how to obtain and preprocess each dataset are mentioned in the readme file of `preprocess` folder 

## Train
The `train_lha.py` can be used to train all the models by setting different arguments.  

###  FOR HGCLR 
`python train_lha.py --name='ckp_hgclr' --batch 10 --data='wos' --graph 1 --lamb 0.05 --thre 0.02` </br>
</br>
Some Important arguments: </br>
- `--lamb` and `--thre` are arguments specific to HGCLR, and their specific values for WOS dataset are given in [contrastive-htc](https://github.com/wzh9969/contrastive-htc#reproducibility)

### FOR LHA-CON
`python train_lha.py --name='ckpt_con' --batch 10 --data='wos' --graph 1 --lamb 0.05 --thre 0.02 --hsampling 1 --hcont_wt 0.4` </br>
</br>
Some Important arguments: </br>
- `--hsampling` whether to use LHA-CON module
-  `--hcont_wt` weight term of the LHA-CON module.

### FOR LHA-ADV
`python train_lha.py --name='ckpt_adv' --batch 10 --data='wos' --graph 1 --lamb 0.05 --thre 0.02 --label_reg 1 --prior_wt 0.5 --hlayer 900` </br> </br>
Some Important arguments: </br>
- `--label_reg` whether to use LHA-ADV
-  `--prior_wt` weight term of the LHA-ADV module
-  `--hlayer` The size of the first  hidden layer of the neural network

### For BERT 

`python train_lha.py --name='ckp_bert' --batch 10 --data='wos' --graph 0` </br> </br>
Some Important arguments: </br>
- `--name` The name of directory in which your model will be saved. For e.g. the above model will be saved in `./LHA-HTC/data/wos/ckp_bert`
- `--data` The name of directory which contains your data and related files
- `--graph` whether to use structure encoder

## Test
To run the trained model on test set run the script `test_lha.py` </br> 
`python test_lha.py --name ckpt1 --data wos --extra _macro` </br> </br>
Some Important arguments
- `--name` The name of the directory which contains the saved checkpoint. The checkpoint is saved in `../LHA-HTC/data/wos/`
- `--data` The name of directory which contains your data and related files
- `--extra` Two checkpoints are kept based on macro-F1 and micro-F1 respectively. The possible choices are  `_macro` and `_micro` to choose from the two checkpoints

## Citation
If you find our work helpful, please cite it using the following BibTeX entry:
```bibtex
@INPROCEEDINGS{10386495,
  author={Kumar, Ashish and Toshniwal, Durga},
  booktitle={2023 IEEE International Conference on Big Data (BigData)}, 
  title={Label Hierarchy Alignment for Improved Hierarchical Text Classification}, 
  year={2023},
  volume={},
  number={},
  pages={1174-1179},
  doi={10.1109/BigData59044.2023.10386495}}
}


