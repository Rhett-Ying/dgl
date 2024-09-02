# New Cache

```
DGL_HOME=/home/ubuntu/workspace/dgl_4 DGL_LIBRARY_PATH=$DGL_HOME/build PYTHONPATH=$DGL_HOME/python:$PYTHONPATH python node_classification.py --gpu-cache-size-in-gigabytes=0 --cpu-cache-size-in-gigabytes=10 --dataset=ogbn-arxiv --epochs=2 --cache-version=2 --mode=cpu-cpu-cuda
```

----------- CPUFeatureCache2 is used ----------- 
Training: 88it [07:30,  5.12s/it, num_nodes=82459, gpu_cache_miss=1, cpu_cache_miss=0]
Evaluating: 30it [02:25,  4.84s/it, num_nodes=79905, gpu_cache_miss=1, cpu_cache_miss=0]
Epoch 00, Loss: 1.8516, Approx. Train: 0.5038, Approx. Val: 0.6447, Time: 450.6125400066376s
Training: 88it [07:29,  5.10s/it, num_nodes=81949, gpu_cache_miss=1, cpu_cache_miss=0]       
Evaluating: 30it [02:25,  4.85s/it, num_nodes=80155, gpu_cache_miss=1, cpu_cache_miss=0]    
Epoch 01, Loss: 1.2522, Approx. Train: 0.6309, Approx. Val: 0.6704, Time: 449.1136426925659s

# Old Cache

```
DGL_HOME=/home/ubuntu/workspace/dgl_4 DGL_LIBRARY_PATH=$DGL_HOME/build PYTHONPATH=$DGL_HOME/python:$PYTHONPATH python node_classification.py --gpu-cache-size-in-gigabytes=0 --cpu-cache-size-in-gigabytes=10 --dataset=ogbn-arxiv --epochs=2 --cache-version=1 --mode=cpu-cpu-cuda
```

Training: 88it [00:12,  7.00it/s, num_nodes=81760, gpu_cache_miss=1, cpu_cache_miss=0.0268]
Evaluating: 30it [00:01, 19.60it/s, num_nodes=79747, gpu_cache_miss=1, cpu_cache_miss=0.0181]
Epoch 00, Loss: 1.9049, Approx. Train: 0.4853, Approx. Val: 0.6457, Time: 12.57881784439087s
Training: 88it [00:04, 21.45it/s, num_nodes=82339, gpu_cache_miss=1, cpu_cache_miss=0.0107]
Evaluating: 30it [00:01, 21.89it/s, num_nodes=79840, gpu_cache_miss=1, cpu_cache_miss=0.00896]
Epoch 01, Loss: 1.2569, Approx. Train: 0.6326, Approx. Val: 0.6718, Time: 4.102634429931641s
