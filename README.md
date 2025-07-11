<!-- PROJECT LOGO -->

<p align="center">
  <h1 align="center">Segment Any 3D Object with Language</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/seungjun-lee-43101a261/">Seungjun Lee</a><sup>*</sup></span> · 
    <a href="https://yuyangzhao.com">Yuyang Zhao</a><sup>*</sup> · 
    <a href="https://www.comp.nus.edu.sg/~leegh/">Gim Hee Lee</a><sup></sup> <br>
    National University of Singapore<br>
    <sup>*</sup>equal contribution
  </p>
  <h2 align="center">ICLR 2025</h2>
  <h3 align="center"><a href="https://github.com/CVRP-SOLE/SOLE">Code</a> | <a href="https://arxiv.org/abs/2404.02157">Paper</a> | <a href="https://cvrp-sole.github.io">Project Page</a> </h3>
  <div align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
  <a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
  </div>
</p>


## About
This repository is for evaluation of SOLE on Replica dataset. Our SOLE is trained on ScanNet200 dataset and evaluated on Replica dataset. More details are on Sec. 4.2 of the main paper.

## How to use it?

First, download the below data:
* <a href="https://huggingface.co/datasets/onandon/SOLE/tree/main/processed/replica">Pre-processed Replica</a>
* <a href="https://huggingface.co/datasets/onandon/SOLE/tree/main/openseg/replica">Precomputed per-point CLIP features of Replica</a>
* <a href="https://huggingface.co/datasets/onandon/SOLE/blob/main/replica.ckpt">Pretrained weight</a>

Move the above files to designated locations:
* Pre-processed Replica → `data/processed/replica/`
* Precomputed per-point CLIP features of Replica → `openseg/replica/`
* Pretrained weight -> `checkpoint/`

Now, you are ready to evaluate our SOLE! Run the below script.

```
bash eval.sh
```

Due to the stochastic nature of voxelization, the performance may exhibit slight variations (up to ±1.0 AP) across different runs.