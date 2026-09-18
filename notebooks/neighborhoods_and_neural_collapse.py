from mlxp import models, training
from mlxp.models import get_embeddings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def make_dataset(
    n_samples: int,
    centers: np.ndarray,
    cluster_std: float,
    random_state: int,
):

    def _calculate_pertinences(
        x: np.ndarray,
        c: np.ndarray,
    ):
        dist = pairwise_distances(x, c, metric="euclidean")
        pert = 1 / dist
        pert /= pert.sum(axis=1, keepdims=True)
        return pert

    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )

    pert = _calculate_pertinences(X, centers)

    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        X,
        y,
        pert,
        test_size=0.2,
        random_state=random_state,
    )

    r = [
        torch.tensor(a).float() for a in [X_train, X_test, y_train, y_test, p_train, p_test]
    ]
    r[2] = r[2].long()
    r[3] = r[3].long()
    return (r_ for r_ in r)


def experiment():
    centers = np.array(
        [
            [1, 1],
            [-1, 1],
            [1, -1],
            [-1, -1],
            [0, 0],
        ]
    )

    X_train, X_test, y_train, y_test, p_train, p_test = make_dataset(
        n_samples=1000,
        centers=centers,
        cluster_std=0.7,
        random_state=42,
    )

    clf = models.MLP(
        input_dim=centers.shape[1],
        hidden_dim=100,
        output_dim=centers.shape[0],
        n_layers=20,
        dropout=0.0,
        block_type="simple",  # "simple" or "residual"
    )

    train_data = training.train_mlp(
        model=clf,
        x=X_train,
        y=y_train,
        p=y_train,
        optimizer_type="adam",
        device="cuda:0",
        batch_size=500,
        val_split=0.2,
        epochs=100,
        seed=42,
        weight_decay=5e-4,
        loss_fn=nn.CrossEntropyLoss(),
        patience=1000000,
        verbose=False,
        lr=1e-3,
        task="classification",
        record_nc=True,
    )

    plt.figure(figsize=(7, 7))
    
    plt.subplot(3,1,1)
    plt.plot(train_data["train_loss"], "b", label="Train loss")
    plt.plot(train_data["val_loss"], "r", label="Val loss")
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.plot(np.array(train_data["train_acc"]), "b", label="Train acccuracy")
    plt.plot(np.array(train_data["val_acc"]), "r", label="Val accuracy")
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(3,1,3)
    plt.plot(train_data['train_nc1'], 'b--', label='Train NC1')
    plt.plot(train_data['val_nc1'], 'r--', label='Val NC1')
    plt.ylabel("NC1")
    plt.xlabel("Epochs")
    plt.legend()
    plt.semilogy()
    
    plt.savefig("training.png")

    z = get_embeddings(model=clf, x=X_train)

    pca_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=2,
                ),
            ),
        ]
    )

    tsne_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "tsne",
                TSNE(
                    n_components=2,
                    perplexity=50,
                ),
            ),
        ]
    )

    z_pca = pca_pipeline.fit_transform(z)
    z_tsne = tsne_pipeline.fit_transform(z)
    
    y = y_train.detach().cpu().numpy()
    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1)
    plt.scatter(X_train[:,0], X_train[:,1], c=y, cmap='jet')
    plt.title('Original data')
    plt.subplot(1,3,2)
    plt.scatter(z_pca[:,0], z_pca[:,1], c=y, cmap='jet')
    plt.title('Embeddings (PCA)')
    plt.subplot(1,3,3)
    plt.scatter(z_tsne[:,0], z_tsne[:,1], c=y, cmap='jet')
    plt.title('Embeddings (TSNE)')
    plt.savefig('embeddings.png')
    
if __name__ == "__main__":
    experiment()