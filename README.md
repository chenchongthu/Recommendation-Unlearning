# RecEraser

This is our implementation of the paper: 

*Chong Chen, Fei Sun, Min Zhang and Bolin Ding. 2022. [Recommendation Unlearning.](https://arxiv.org/pdf/2201.06820.pdf) 
In TheWebConf'22.*

**Please cite our TheWebConf'22 paper if you use our codes. Thanks!**

```
@inproceedings{chen2022recommendation,
  title={Recommendation Unlearning},
  author={Chen, Chong and Sun, Fei and Zhang, Min and Ding, Bolin},
  booktitle={Proceedings of The Web Conference},
  year={2022},
}
```

Author: Chong Chen (cstchenc@163.com)

# C++ evaluator

We use C++ code to output metrics during and after training, as used in [LightGCN](https://github.com/kuandeng/LightGCN), which is much more efficient than python evaluator. It needs to be compiled first using the following command:
```
python setup.py build_ext --inplace
```
After compilation, the C++ code will run by default instead of Python code.

# Balanced Data Partition

The code of data partition is in code/utility/data_partition.py.

The pre-train embedding vectors are computed by WMF in this work.

# Hype-Parameters

The instruction of commands has been stated in the codes (see the parser function in code/utility/parser.py).

The hype-parameters for base models are:

```
yelp2018:
BPR: adagrade	lr=0.05	reg=0.01	batch=256
WMF: adagrade	lr=0.05	reg=0.01	batch=256	weight=0.05	drop=0.7
LightGCN: adam	lr=0.001	reg=1e-4	batch=1024

ml-1m:
BPR: adagrade	lr=0.05	reg=0.01	batch=256
WMF: adagrade	lr=0.05	reg=0.01	batch=256	weight=0.2	drop=0.7
LightGCN: adam lr=0.001	reg=1e-3	batch=1024

ml-10m:

BPR: adagrade	lr=0.05	reg=0.001	batch=256
WMF: adagrade	lr=0.05	reg=0.01	batch=256	weight=0.2	drop=0.7
LightGCN: adam lr=0.001	reg=1e-3	batch=1024
```

















