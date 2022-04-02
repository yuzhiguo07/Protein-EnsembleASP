# [Protein-EnsembleASP] Deep Ensemble Learning with Atrous Spatial Pyramid Networks for Protein Secondary Structure Prediction
The secondary structure of proteins is significant for studying the three-dimensional structure and functions of proteins. Several models from image understanding and natural language modeling have been successfully adapted in the protein sequence study area, such as Long Short-term Memory (LSTM) network and Convolutional Neural Network (CNN). Recently, Gated Convolutional Neural Network (GCNN) has been proposed for natural language processing. It has achieved high levels of sentence scoring, as well as reduced the latency. Conditionally Parameterized Convolution (CondConv) is another novel study which has gained great success in the image processing area. Compared with vanilla CNN, CondConv uses extra sample-dependant modules to conditionally adjust the convolutional network. In this paper, we propose a novel Conditionally Parameterized Convolutional network (CondGCNN) which utilizes the power of both CondConv and GCNN. CondGCNN leverages an ensemble encoder to combine the capabilities of both LSTM and CondGCNN to encode protein sequences by better capturing protein sequential features. In addition, we explore the similarity between the secondary structure prediction problem and the image segmentation problem, and propose an ASP network (Atrous Spatial Pyramid Pooling~(ASPP) based network) to capture fine boundary details in secondary structure. Extensive experiments show that the proposed method can achieve higher performance on protein secondary structure prediction task than existing methods on CB513, Casp11 and CASP12 datasets. We also conducted ablation studies over each component to verify the effectiveness. Our method is expected to be useful for any protein related prediction tasks, which is not limited to protein secondary structure prediction.

# Userguide
## preparation
Download Uniref50 fasta format database from https://www.uniprot.org/downloads to `./db/uniref50.fasta`

Donwload HMMER3.2.1 from http://hmmer.org/download.html and install the HMMER following the Userguide.pdf.

## Requirements:
cuda 10.2

python 3.6

pytorch 1.4.0

smile `pip install smile`

## Function
### Run jackhmmer
Run Jackhmmer following the Jackhmmer guide book(http://eddylab.org/software/hmmer/Userguide.pdf), and fit the output to the `aln_example/sample.aln` format. Here are the hmmer parameters:
```
phmmer -E 1 --domE 1 --incE 0.01 --incdomE 0.03 --mx BLOSUM62 --pextend 0.4 --popen 0.02 -o {out_path} -A {sto_path} --notextw --cpu {cpu_num} {fasta_path} {db_path}
```

### Calculate PSSM
```
python calculate_pssm.py --aln_path ./aln_example/sample.aln --save_path ./feat_example/sample.pssm --method 1 
```
#### Parameters
*aln_path* - MSA file path

*save_path* - save pssm feature file path

*method* - PSSM calculation method num, `0`, `1` or `2`, Usually using `1`.

*ss_path*(optional) - secondary structure label file path, see `./ss_example/sample.ss` for an example.
## Upcoming
The `model_condgated_cbe_asp.py` is the core part of the EnsembleASP network. 
We are working on the code cleaning and we will check in all the code regarding the entire pipeline.
