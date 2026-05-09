# Project Folder Structure (full tree)

> Full tree of the modeling phase. The deployment phase adds `deployment/` (see [deployment/README.md](../../deployment/README.md) for that subtree).

Each model subfolder contains: pipeline notebook/script, README with framework notes/time estimates, results (plots/metrics), and data loading consistent with root guidelines.

```text
├── README.md
├── LICENSE
├── data/                                       # .gitignore'd (raw datasets + preprocessed arrays + comparison JSONs)
│   ├── raw/                                    # Source datasets + RL env render frames
│   │   ├── vehicles.csv
│   │   ├── creditcard.csv
│   │   ├── bank-additional-full.csv
│   │   └── eda_envs/                           # Q-Learning #20: env render frames + reward-distribution PNG
│   ├── processed/                              # Preprocessed numpy arrays per model (no q_learning/ - RL is online)
│   │   ├── linear_regression/
│   │   ├── logistic_regression/
│   │   ├── knn/
│   │   ├── kmeans/
│   │   ├── naive_bayes_gaussian/
│   │   ├── naive_bayes_text/
│   │   ├── decision_tree/
│   │   ├── svm/
│   │   ├── pca/
│   │   ├── dnn/
│   │   ├── autoencoder/
│   │   ├── cnn/
│   │   ├── rnn/
│   │   ├── lstm/                               # ECG5000 (augmented) + IMDB (padded sequences)
│   │   ├── gans/                               # CIFAR-10, [-1, 1] normalized for tanh output
│   │   ├── attention/                          # Tatoeba EN-ES, word-level vocab + splits
│   │   ├── transformers_translation/           # Tatoeba BPE 8K shared EN+ES
│   │   ├── transformers_classification/        # AG News BPE 16K English
│   │   ├── gnn/                                # Cora (BoW row-norm) + ogbn-arxiv (symmetrized)
│   │   └── vae/                                # MNIST + CIFAR-10, [0, 1] normalized
│   └── results/                                # Cross-framework comparison JSONs (one per model/dataset)
│       ├── kmeans.json
│       ├── naive_bayes.json
│       ├── decision_tree.json
│       ├── svm.json
│       ├── pca.json
│       ├── dnn.json
│       ├── autoencoder.json
│       ├── cnn.json
│       ├── rnn.json
│       ├── lstm_ecg.json
│       ├── lstm_imdb.json
│       ├── gans.json
│       ├── attention.json
│       ├── transformers_translation.json
│       ├── transformers_classification.json
│       ├── vit.json
│       ├── gnn_cora.json
│       ├── gnn_ogbn_arxiv.json
│       ├── vae_mnist.json
│       ├── vae_cifar10.json
│       ├── q_learning_taxi.json
│       ├── q_learning_cartpole.json
│       └── q_learning_lunarlander.json
├── data-preperation/                           # Preprocess scripts + EDA notebooks per model (#01 -> #20)
│   ├── clean_vehicles.py                       # #01 Linear Regression
│   ├── preprocess_logistic.py                  # #02 Logistic Regression
│   ├── preprocess_knn.py                       # #03 KNN
│   ├── preprocess_kmeans.py                    # #04 K-Means
│   ├── preprocess_naive_bayes.py               # #05 Naive Bayes
│   ├── preprocess_decision_tree.py             # #06 Decision Trees / RF
│   ├── eda_decision_tree.ipynb
│   ├── preprocess_svm.py                       # #07 SVM
│   ├── eda_svm.ipynb
│   ├── preprocess_pca.py                       # #08 PCA
│   ├── eda_pca.ipynb
│   ├── preprocess_dnn.py                       # #09 DNN
│   ├── eda_dnn.ipynb
│   ├── preprocess_autoencoder.py               # #10 Autoencoders
│   ├── eda_autoencoder.ipynb
│   ├── preprocess_cnn.py                       # #11 CNN
│   ├── eda_cnn.ipynb
│   ├── preprocess_rnn.py                       # #12 RNN
│   ├── eda_rnn.ipynb
│   ├── preprocess_lstm.py                      # #13 LSTM
│   ├── eda_lstm.ipynb
│   ├── preprocess_gans.py                      # #14 GANs
│   ├── eda_gans.ipynb
│   ├── preprocess_attention.py                 # #15 Attention
│   ├── eda_attention.ipynb
│   ├── preprocess_transformers_translation.py  # #16 Transformers (2 datasets: translation + classification)
│   ├── preprocess_transformers_classification.py
│   ├── eda_transformers_translation.ipynb
│   ├── eda_transformers_classification.ipynb
│   ├── eda_vit.ipynb                           # #17 ViT (no preprocess; reused CNN #11 data)
│   ├── preprocess_gnn.py                       # #18 GNN
│   ├── eda_gnn.ipynb
│   ├── preprocess_vae.py                       # #19 VAE
│   ├── eda_vae.ipynb
│   └── eda_envs.ipynb                          # #20 Q-Learning (no preprocess; RL is online)
├── utils/                                      # Shared utilities (#02 onward; see docs/modeling/utilities.md)
│   ├── __init__.py
│   ├── data_loader.py
│   ├── metrics.py
│   ├── performance.py
│   ├── visualization.py
│   ├── results.py
│   ├── tree_utils.py
│   ├── svm_utils.py
│   ├── rnn_utils.py
│   ├── gan_utils.py
│   ├── attention_utils.py
│   ├── transformer_utils.py
│   ├── vit_utils.py
│   ├── gnn_utils.py
│   ├── vae_utils.py
│   └── rl_utils.py
├── No-Framework/                               # #01-#08 only (retired after PCA; from-scratch ceiling for classical ML)
│   ├── 01-linear-regression/
│   ├── 02-logistic-regression/
│   ├── 03-knn/
│   ├── 04-k-means/
│   ├── 05-naive-bayes/
│   ├── 06-decision-trees-random-forests/
│   ├── 07-svm/
│   └── 08-pca/
├── Scikit-Learn/                               # #01-#10 only (retired after Autoencoders; sklearn MLP can't compete with PT/TF conv AE)
│   ├── 01-linear-regression/
│   ├── 02-logistic-regression/
│   ├── 03-knn/
│   ├── 04-k-means/
│   ├── 05-naive-bayes/
│   ├── 06-decision-trees-random-forests/
│   ├── 07-svm/
│   ├── 08-pca/
│   ├── 09-dnn/
│   └── 10-autoencoders/
├── PyTorch/                                    # All 20 models; deployment winner for 14 of 17 deployable artifacts
│   ├── 01-linear-regression/
│   ├── 02-logistic-regression/
│   ├── 03-knn/
│   ├── 04-k-means/
│   ├── 05-naive-bayes/
│   ├── 06-decision-trees-random-forests/
│   ├── 07-svm/
│   ├── 08-pca/
│   ├── 09-dnn/
│   ├── 10-autoencoders/
│   ├── 11-cnn/
│   ├── 12-rnn/
│   ├── 13-lstm/
│   ├── 14-gans/
│   ├── 15-attention/
│   ├── 16-transformers/
│   ├── 17-vit/
│   ├── 18-gnn/
│   ├── 19-vae/
│   └── 20-q-learning/
├── TensorFlow/                                 # All 20 models with WSL2 cuDNN scope reductions on #17 V3, #18 V2/V4, #19 V2, #20 V3-V5
│   ├── 01-linear-regression/
│   ├── 02-logistic-regression/
│   ├── 03-knn/
│   ├── 04-k-means/
│   ├── 05-naive-bayes/
│   ├── 06-decision-trees-random-forests/
│   ├── 07-svm/
│   ├── 08-pca/
│   ├── 09-dnn/
│   ├── 10-autoencoders/
│   ├── 11-cnn/
│   ├── 12-rnn/
│   ├── 13-lstm/
│   ├── 14-gans/
│   ├── 15-attention/
│   ├── 16-transformers/
│   ├── 17-vit/
│   ├── 18-gnn/
│   ├── 19-vae/
│   └── 20-q-learning/
└── deployment/                                 # ACTIVE: Production FastAPI services for D1-D5 deployment winners
    ├── README.md                               # Phase progress + architecture + per-service links
    ├── docs/
    ├── scripts/                                # promote_to_registry.py, verify_artifacts.py
    ├── services/                               # 3 services: sklearn-svc, pt-svc, tf-svc
    │   ├── sklearn-svc/                        # D1 SK PCA 
    │   ├── pt-svc/                             # D2 PT DNN, D3 GAN, D4 Q-Learning (planned)
    │   └── tf-svc/                             # D5 TF Transformer Translation (planned)
    ├── mlflow.db                               # consolidated MLflow registry (gitignored)
    └── mlruns/                                 # registry artifact store (gitignored)
```
