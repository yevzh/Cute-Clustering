# README

This is the first homework of Machine Learning in the spring semester of 2023.

The code repository consists mainly of two tasks:

- RPCL algorithm for K-means

  please refer to `rpclk.py` and `RPCLK4KM.py`

  ```bash
  python rpclk.py --k x --beta x
  ```

- model selections for GMM clustering

  please refer to `GMM.py` and `model_selection.py`

  ```bash
  python GMM.py --n_components x --n_clusters x --n_samples x --cluster_std x --method x
  ```

The implementation mainly focuses on how to automatically get the best k.

## UPDATED!!!

To do the experiments, you can directly use the bash command:

```bash
bash experiment_rpclk.sh
```

```bash
bash experiment_gmm.sh
```





