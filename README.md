# VGG16_TransferLearning
1.datasets from kaggle:https://www.kaggle.com/c/dogs-vs-cats/data

		
2.result figure:

# How to do 
1.use import_small_dataset.py to spilt data to train_new , validation , test files.
2.Then run extract_features.py


# Problems
train loss 不断下降，test loss不断下降，说明网络仍在学习;

train loss 不断下降，test loss趋于不变，说明网络过拟合;

train loss 趋于不变，test loss不断下降，说明数据集100%有问题;

train loss 趋于不变，test loss趋于不变，说明学习遇到瓶颈，需要减小学习率或批量数目;

train loss 不断上升，test loss不断上升，说明网络结构设计不当，训练超参数设置不当，数据集经过清洗等问题。



# Environment
My environment is based on

ubuntu16.04 LTS

Memory: 15.6GiB

Processor: Intel® Core™ i7-7700 CPU @ 3.60GHz × 8 

Graphics:GeForce GTX 1070 Ti/PCIe/SSE2

OS type：64-bit

Disk:475.3GB

NVIDIA-SMI 440.33.01    
Driver Version: 440.33.01    
CUDA Version: 10.2

