# NeoMUST

## Introduction

NeoMUST: an Accurate and Efficient Multi-Task Learning Model for Neoantigen Presentation

Contract: hui.yao@freshwindbiotech.com

## Installation

There are two ways to install NeoMUST.

### 1. Docker (Recommend)

The Installation of Docker can be seen in https://docs.docker.com/

Pull the image of neomust from dockerhub:

    docker pull freshwindbioinformatics/neomust:v1

Run the image in bash:

    docker run -it --gpus all freshwindbioinformatics/neomust:v1 bash

####  * Note : The parameter "--gpus" requires docker version higher than 19.03.

### 2. Conda and pip

#### Dependencies

* python == 3.9.12
* mhcnames == 0.4.8
* numpy == 1.21.5
* pandas == 1.2.0
* torch == 1.11.0

####  * Note : If you want to use the GPU, you should install CUDA and cuDNN version compatible with the pytorch version. [Version Searching](https://pytorch.org/)

Command:

    conda create -n neomust python==3.9.12
    conda activate neomust
    pip install -r ./requirements.txt

####  * Note : How to download and install conda? [Documentation](https://docs.conda.io/en/latest/miniconda.html).

## Usage

Using NeoMUST for Prediction and Training NeoMUST with Your Own Data.

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
          --batch_size INT: batch_size in pytorch Dataloader (default: 1024)
          --num_workers INT: num_workers in pytorch Dataloader (default: 0)
          --pin_memory BOOL: pin_memory in pytorch Dataloader (default: False)
          --max_len INT: Maximum length per task to be split (default: 50000)
          --max_task INT: Maximum number of parallel tasks (default: 2)

Command:

    cd NeoMUST  # If using docker, run this line first.
    python ./Neomust/predict_cl.py --input ./Test/demo_data.csv --blosum62 ./Data/BLOSUM62.txt --mhc_aa ./Data/allele_sequences.csv --neomust_model ./Neomust/model/neomust_model.pt --rank_database ./Data/rank_database_lite --output ./Test/output.csv

#### Input

NeoMUST uses a **csv** file as input, with the header including **hla** and **peptide**. (required)

For example (Test/demo_data.csv):

    sample_id,hla,peptide,hit
    3,A*01:01,ASSFLKSFY,0
    3,A*01:01,ATLFSDSWYY,0
    3,A*01:01,CSDSGKSFINY,0
    2,A*01:01,CVDWLIAVY,0
    3,A*01:01,DTDSRFISY,0
    2,A*01:01,ELTQGYIYFY,0
    ...

#### Output

NeoMUST takes a **csv** file as output, with the header including **neomust_ba**, **neomust_el** and **neomust_el_rank**. 
In the article, we use neomust_ba in TeSet-3, and neomust_el_rank in TeSet-1, TeSet-2, TeSet-1-Filtered, TeSet-2-Filtered and TeSet-4.

For example (Test/output.csv):

    sample_id,hla,peptide,hit,neomust_ba,neomust_el,neomust_el_rank
    3,A*01:01,ASSFLKSFY,0,132.2011260986328,0.9949237704277039,0.0188624538908844
    3,A*01:01,ATLFSDSWYY,0,153.15562438964844,0.9936830997467041,0.0214759749359707
    3,A*01:01,CSDSGKSFINY,0,44.277076721191406,0.9998539686203003,0.00157858281922
    2,A*01:01,CVDWLIAVY,0,149.58494567871094,0.9954954385757446,0.0176020038337011
    3,A*01:01,DTDSRFISY,0,19.983366012573242,0.9998342990875244,0.0018242296354762
    2,A*01:01,ELTQGYIYFY,0,446.3338317871094,0.9117483496665955,0.095109614858008
    ...

neomust_ba : BA predicted by the NeoMUST.

neomust_el : EL predicted by the NeoMUST.

neomust_el_rank : Rank of EL predicted by the NeoMUST in the rank database.

### Train

    Usage: train_cl.py [options]
    Required:
          --input STRING: The input file to be trained (*.csv) 
                          Required columns: "allele", "peptide", "af", "measurement_inequality", "measurement_kind", "ms" 
          --blosum62 STRING: The BLOSUM62 file (*.txt)
          --mhc_aa STRING: The MHC_pseudo-sequences file (*.csv)
                           Required columns: "allele", "sequence"
          --output_path STRING: The output path

    Optional:
          --batch_size INT: batch_size in pytorch Dataloader (default: 1024)
          --pin_memory BOOL: pin_memory in pytorch Dataloader (default: False)
          --max_epochs INT: Maximum epochs (default: 60)

Command:

    python ./Neomust/train_cl.py --input ./Train/demo_data.csv --blosum62 ./Data/BLOSUM62.txt --mhc_aa ./Data/allele_sequences.csv --output_path ./Train

#### Input

NeoMUST uses a **csv** file as input, with the header including **hla**, **peptide**, **af**, **measurement_inequality**, **measurement_kind** and **ms**. (required)

For example (Train/demo_data.csv):

    hla,peptide,af,measurement_inequality,measurement_kind,ms
    HLA-A*01:01,AAGLPAIFV,5000,>,affinity,-1
    HLA-A*01:01,AASGFTFSSY,4972.665793,=,affinity,-1
    HLA-B*27:05,AAAKDSHEDHDTSTE,30000,>,mass_spec,0
    HLA-B*27:02,LMLVAGCS,30000,>,mass_spec,0
    HLA-A*02:01,AAIEASQSL,100,<,mass_spec,1
    HLA-A*03:01,AAFGGTFKK,100,<,mass_spec,1
    ...

#### Output

Model (epoch*.pt) and loss information (history.csv) for each epoch.

For example (Train/history.csv):
    
    epoch,loss_af,loss_ms,val_loss_af,val_loss_ms,coef_var_af,coef_var_ms
    1.0,0.03764456720233084,0.23564889253273938,0.02382027448408983,0.17882955277507956,0.5289607048034668,0.7348414659500122
    2.0,0.023393643075415135,0.17452709180346368,0.021120972499590027,0.16742397980256515,0.3363616168498993,0.705833375453949
    3.0,0.020979259975451,0.15976523586811447,0.018658501417799428,0.1448341972448609,0.3258369266986847,0.6899664402008057
    4.0,0.0198006893358898,0.151081842670697,0.018063104364343666,0.13959771103479646,0.3216213881969452,0.6802341341972351
    5.0,0.01900767592777466,0.14508853885996897,0.01741925247267566,0.1348600729622624,0.3183627128601074,0.674423336982727
    6.0,0.018541027978818268,0.14056530609938483,0.017232906572859395,0.13296843421730128,0.3162034749984741,0.6678964495658875
    ...

## Note

Due to space limitation, we have created rank_database_lite for benchmark and test data, the full version of
rank_database can be downloaded from the docker image.

Command:

    docker pull freshwindbioinformatics/neomust:v1
    docker cp -r neomust:/workspace/Data/rank_database_lite /path/on/host
