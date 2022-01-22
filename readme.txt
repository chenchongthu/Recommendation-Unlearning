1、使用WMF预训练模型，产生向量
2、parse.py 调节参数

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
LightGCN: lr=0.001	reg=1e-3	batch=1024

3、用c计算评价指标，速度更快
