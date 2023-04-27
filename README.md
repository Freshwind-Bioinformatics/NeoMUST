# NeoMUST

## Introduction

NeoMUST: an Accurate and Efficient Multi-Task Learning Model for Neoantigen Presentation

Contract: hui.yao@freshwindbiotech.com

## Requirements

* python == 3.9.12
* mhcnames == 0.4.8
* numpy == 1.21.5
* pandas == 1.2.0
* torch == 1.11.0

####   * Note : If you want to use the GPU, you should install CUDA and cuDNN version compatible with the pytorch version. [Version Searching](https://pytorch.org/)

## Installation

Command:

    conda create -n neomust python==3.9.12
    conda activate neomust
    pip install -r ./requirements.txt

####   * Note : How to download and install conda? [Documentation](https://docs.conda.io/en/latest/miniconda.html).

## Usage

### Predict

    Usage: predict_cl.py [options]
    Required:
          --input STRING: The input file to be predicted (*.csv) 
                          Required columns: "hla", "peptide" 
          --blosum62 STRING: The BLOSUM62 file (*.txt)
          --mhc_aa STRING: The MHC_pseudo-sequences file (*.csv)
                           Required columns: "allele", "sequence" 
          --neomust_model STRING: The trained NeoMUST model file (*.pt)
          --rank_database STRING: The Rank database path (/path)
          --output STRING: The output file (*.csv)

    Optional:
          --batch_size INT: batch_size in pytorch Dataloader (default: 2048)
          --num_workers INT: num_workers in pytorch Dataloader (default: 0)
          --pin_memory BOOL: pin_memory in pytorch Dataloader (default: False)
          --max_len INT: Maximum length per task to be split (default: 50000)
          --max_task INT: Maximum number of parallel tasks (default: Number of cores in your CPU)

Command:

    python predict_cl.py --input ./Test/demo_data.csv --blosum62 ./Data/BLOSUM62.txt --mhc_aa ./Data/allele_sequences.csv --neomust_model ./Neomust/model/neomust_model.pt --rank_database ./Data/rank_database_lite --output ./Test/output.csv

## Input

NeoMUST uses a **csv** file as input, with the header including **hla** and **peptide**. (required)

For example (test/demo_data.csv):

    sample_id,hla,peptide,hit
    3,A*01:01,ASSFLKSFY,0
    3,A*01:01,ATLFSDSWYY,0
    3,A*01:01,CSDSGKSFINY,0
    2,A*01:01,CVDWLIAVY,0
    3,A*01:01,DTDSRFISY,0
    2,A*01:01,ELTQGYIYFY,0
    ...

## Output

NeoMUST takes a **csv** file as output, with the header including **neomust_ba**, **neomust_el** and **neomust_el_rank**
.

For example (test/output.csv):

    sample_id,hla,peptide,hit,neomust_ba,neomust_el,neomust_el_rank
    3,A*01:01,ASSFLKSFY,0,234.1349334716797,0.5148767232894897,0.2155490407693175
    3,A*01:01,ATLFSDSWYY,0,237.1898651123047,0.5150091648101807,0.2155208517904028
    3,A*01:01,CSDSGKSFINY,0,231.74472045898438,0.5153178572654724,0.2154161498687199
    2,A*01:01,CVDWLIAVY,0,235.4841766357422,0.5167423486709595,0.2150496931428295
    3,A*01:01,DTDSRFISY,0,238.1300048828125,0.5167102217674255,0.2150577471368051
    2,A*01:01,ELTQGYIYFY,0,235.18215942382812,0.5166894793510437,0.2150617741337929
    ...

neomust_ba : BA predicted by the NeoMUST.

neomust_el : EL predicted by the NeoMUST.

neomust_el_rank : Rank of EL predicted by the NeoMUST in the rank database.

## Note

Due to space limitation, we have created rank_database_lite for benchmark and test data, the full version of
rank_database can be downloaded from (link).