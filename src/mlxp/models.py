from itertools import pairwise

import torch
from torch import nn


class SimpleBlock(nn.Module):
    """A simple MLP block: Linear -> ReLU -> Dropout."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    """He, Kaiming; Zhang, Xiangyu; Ren, Shaoqing; Sun, Jian (2016).
    Deep Residual Learning for Image Recognition.
    Conference on Computer Vision and Pattern Recognition.
    arXiv:1512.03385. doi:10.1109/CVPR.2016.90.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.batch_norm2 = nn.BatchNorm1d(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.batch_norm2(x)
        x += residual
        return x


class MLP(nn.Module):
   

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        block_type: str = "simple",  # "simple" or "residual"
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_batch_norm = nn.BatchNorm1d(hidden_dim)

        if block_type == "simple":
            block_class = SimpleBlock
        elif block_type == "residual":
            block_class = ResidualBlock
        else:
            raise ValueError(f"Unknown block_type: {block_type}")
        
        self.hidden_layers = nn.ModuleList(
            block_class(hidden_dim, hidden_dim, dropout) for _ in range(n_layers)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the hidden representation fed into the output layer.

        This is the "embedding" used for point-cloud similarity
        comparisons between models (see `get_embeddings`).
        """
        x = self.input_layer(x)
        x = self.input_batch_norm(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.forward_features(x))

class EmbeddingMLP(nn.Module):
    """Embeds a set of categorical features and feeds their concatenation into an MLP.

    Each categorical column gets its own learned `nn.Embedding` table;
    the per-column embeddings are concatenated before the MLP body, so
    the network can learn feature interactions the embeddings alone
    can't express, unlike a fixed one-hot encoding.
    """

    def __init__(
        self,
        num_categories: list[int],
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        block_type: str = "simple",
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(n, embedding_dim) for n in num_categories
        )
        self.mlp = MLP(
            input_dim=embedding_dim * len(num_categories),
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            dropout=dropout,
            block_type=block_type,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the hidden representation fed into the output layer."""
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.mlp.forward_features(torch.cat(embedded, dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp.output_layer(self.forward_features(x))


def get_device() -> torch.device:
    """Return the best available device (CUDA GPU if usable, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_embeddings(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device | None = None,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Run `model.forward_features` over `x` in batches, returning a CPU tensor.

    This is the per-row hidden representation ("point cloud") used to
    compare models with `pointcloudsimilarity`. Batched so a full test
    set doesn't need to fit through the model in one forward pass.
    """
    device = device or get_device()
    model.to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            batch = x[start : start + batch_size].to(device)
            outputs.append(model.forward_features(batch).cpu())
    return torch.cat(outputs, dim=0)




if __name__ == "__main__":
    from .training import train_mlp

    device = get_device()
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")

    # Smoke test with random data.
    x = torch.randn(500, 20)
    y = torch.randn(500, 1)

    model = MLP(input_dim=20, hidden_dim=32, output_dim=1, n_layers=2)
    history = train_mlp(model, x, y, epochs=10, batch_size=32, device=device)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
