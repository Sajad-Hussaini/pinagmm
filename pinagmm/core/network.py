"""
Neural network architecture for PINAGMM.

Implements a Physics-Informed Neural Additive Model (NAM) with:
- Monotonic sub-networks (via softplus-constrained weights) for features
  that have known physical monotonic relationships (Mw, Rrup)
- Free sub-networks for all other features
- Pairwise interaction terms
- A global bias term
- Grouped ensemble training (one network per output group)

These classes are used internally during training only.
At inference time, the trained ensemble is loaded from a .joblib file.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .variables import group_definitions


# ---------------------------------------------------------------------------
# Monotonic linear layer
# ---------------------------------------------------------------------------
class MonotonicLinear(nn.Linear):
    """
    Linear layer with non-negative weights (enforced via softplus).

    Ensures the sub-network output is monotonically non-decreasing in its
    inputs (after optional sign-flipping at the architecture level).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, F.softplus(self.weight), self.bias)


# ---------------------------------------------------------------------------
# Per-feature sub-network
# ---------------------------------------------------------------------------
class SubNetwork(nn.Module):
    """
    A compact MLP for one input feature (or one interaction pair).

    Architecture:
        deep path: Linear → Tanh → … → Linear (no bias on last layer)
        skip path: direct Linear from input to output (only for free nets)
        output   : deep + skip   (residual-style, improves gradient flow)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_layers: list[int],
        dropout: float = 0.0,
        is_monotonic: bool = False,
    ):
        super().__init__()
        LinearType = MonotonicLinear if is_monotonic else nn.Linear

        layers = []
        curr = in_dim
        for h in hidden_layers:
            layers.append(LinearType(curr, h))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            curr = h
        layers.append(LinearType(curr, out_dim, bias=False))
        self.deep_path = nn.Sequential(*layers)

        self.is_monotonic = is_monotonic
        if not is_monotonic:
            self.skip = LinearType(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.deep_path(x)
        if not self.is_monotonic:
            out = out + self.skip(x)
        return out


# ---------------------------------------------------------------------------
# Pure additive network (one sub-net per input)
# ---------------------------------------------------------------------------
class PureAdditiveNetwork(nn.Module):
    """
    Neural Additive Model (NAM): output = bias + Σ_i f_i(x_i) + Σ_j g_j(x_inter_j).

    Monotonic and free features are handled by separate sub-network lists.
    Interaction terms are optional pairwise (or higher-order) sub-networks.
    """

    def __init__(
        self,
        mono_in_dim: int,
        free_in_dim: int,
        out_dim: int,
        hidden_layers: list[int],
        dropout: float = 0.0,
        interactions: list = None,
    ):
        super().__init__()

        self.mono_nets = nn.ModuleList([
            SubNetwork(1, out_dim, hidden_layers, dropout, is_monotonic=True)
            for _ in range(mono_in_dim)
        ])
        self.free_nets = nn.ModuleList([
            SubNetwork(1, out_dim, hidden_layers, dropout, is_monotonic=False)
            for _ in range(free_in_dim)
        ])

        self.interactions = interactions or []
        self.interaction_nets = nn.ModuleList([
            SubNetwork(
                in_dim=len(inter["features"]),
                out_dim=out_dim,
                hidden_layers=hidden_layers,
                dropout=dropout,
                is_monotonic=inter.get("monotonic", False),
            )
            for inter in self.interactions
        ])

        self.global_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(
        self,
        x_mono: torch.Tensor | None,
        x_free: torch.Tensor | None,
        x_inter: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        out = self.global_bias
        if x_mono is not None:
            out = out + sum(
                net(x_mono[:, i : i + 1]) for i, net in enumerate(self.mono_nets)
            )
        if x_free is not None:
            out = out + sum(
                net(x_free[:, i : i + 1]) for i, net in enumerate(self.free_nets)
            )
        if x_inter is not None:
            out = out + sum(
                net(x_inter[i]) for i, net in enumerate(self.interaction_nets)
            )
        return out


# ---------------------------------------------------------------------------
# Group-level architecture (wraps PureAdditiveNetwork for one output group)
# ---------------------------------------------------------------------------
class GroupArchitecture(nn.Module):
    """Wraps PureAdditiveNetwork and handles feature selection/sign-flipping."""

    def __init__(
        self,
        input_dim: int,
        out_idx: list,
        mono_in: list,
        signs: list,
        config: dict,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.out_idx = out_idx
        self.device = device
        self.config = config

        mono_in = config.get("mono_features", mono_in)
        signs = config.get("mono_signs", signs)
        interactions = config.get("interactions", [])
        free_in = config.get(
            "free_features", sorted(set(range(input_dim)) - set(mono_in))
        )

        self.register_buffer("mono_idx", torch.tensor(mono_in, dtype=torch.long))
        self.register_buffer("signs_t", torch.tensor(signs, dtype=torch.float32))
        self.register_buffer("free_idx", torch.tensor(free_in, dtype=torch.long))

        self.interactions = interactions
        for i, inter in enumerate(interactions):
            self.register_buffer(
                f"inter_{i}_idx", torch.tensor(inter["features"], dtype=torch.long)
            )
            if inter.get("monotonic", False) and "signs" in inter:
                self.register_buffer(
                    f"inter_{i}_signs",
                    torch.tensor(inter["signs"], dtype=torch.float32),
                )

        self.network = PureAdditiveNetwork(
            mono_in_dim=len(mono_in),
            free_in_dim=len(free_in),
            out_dim=len(out_idx),
            hidden_layers=config["hidden_layers"],
            dropout=config.get("dropout", 0.0),
            interactions=interactions,
        )
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mono = (
            (x[:, self.mono_idx] * self.signs_t) if len(self.mono_idx) > 0 else None
        )
        x_free = x[:, self.free_idx] if len(self.free_idx) > 0 else None

        x_inter: list | None = None
        if self.interactions:
            x_inter = []
            for i, inter in enumerate(self.interactions):
                idx = getattr(self, f"inter_{i}_idx")
                x_i = x[:, idx]
                if inter.get("monotonic", False) and hasattr(self, f"inter_{i}_signs"):
                    x_i = x_i * getattr(self, f"inter_{i}_signs")
                x_inter.append(x_i)

        return self.network(x_mono, x_free, x_inter)

    def compute_l2_penalty(self) -> torch.Tensor:
        """One-sided L2 to prevent dead units and vanishing gradients in MonotonicLinear."""
        l2 = torch.tensor(0.0, device=self.device)
        for name, module in self.named_modules():
            if isinstance(module, (MonotonicLinear, nn.Linear)) and "skip" not in name:
                if hasattr(module, "weight") and module.weight.requires_grad:
                    if isinstance(module, MonotonicLinear):
                        l2 = l2 + torch.sum(F.relu(module.weight) ** 2)
                    else:
                        l2 = l2 + torch.sum(module.weight**2)
        return l2


# ---------------------------------------------------------------------------
# Single-group estimator (pure Python, not nn.Module)
# ---------------------------------------------------------------------------
class SingleGroupGMM:
    """
    Scikit-learn-style estimator wrapping GroupArchitecture for one output group.

    Uses full-batch L-BFGS with optional early stopping on a validation set.
    """

    def __init__(self, input_dim, out_idx, mono_in, signs, config, device="cpu"):
        self.input_dim = input_dim
        self.out_idx = out_idx
        self.mono_in = mono_in
        self.signs = signs
        self.config = config
        self.device = device
        self._arch = None
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []

    def fit(self, X, y, X_val=None, y_val=None):
        """Train with full-batch L-BFGS."""
        self._arch = GroupArchitecture(
            input_dim=self.input_dim,
            out_idx=self.out_idx,
            mono_in=self.mono_in,
            signs=self.signs,
            config=self.config,
            device=self.device,
        )
        self.train_loss_history = []
        self.val_loss_history = []

        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        wd_lambda = self.config.get("weight_decay", 0.0)
        optimizer = optim.LBFGS(
            self._arch.parameters(),
            lr=self.config.get("lr", 1.0),
            max_iter=self.config.get("lbfgs_max_iter", 20),
            history_size=self.config.get("lbfgs_history", 100),
            line_search_fn="strong_wolfe",
        )
        mse_loss = nn.MSELoss()

        has_val = X_val is not None and y_val is not None
        best_state = None
        best_val = float("inf")
        patience_ctr = 0

        if has_val:
            X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
            y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)
            patience = self.config.get("patience", 5)
            best_state = {
                k: v.cpu().clone() for k, v in self._arch.state_dict().items()
            }

        epochs = self.config.get("epochs", 5)
        train_tol = self.config.get("train_tol", 1e-4)
        prev_loss = float("inf")

        for _ in range(epochs):
            self._arch.train()

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss = mse_loss(self._arch(X_t), y_t)
                if wd_lambda > 0:
                    loss = loss + 0.5 * wd_lambda * self._arch.compute_l2_penalty()
                loss.backward()
                return loss

            loss_val = optimizer.step(closure)
            current_loss = loss_val.item()
            self.train_loss_history.append(current_loss)

            if has_val:
                self._arch.eval()
                with torch.no_grad():
                    val_loss = mse_loss(self._arch(X_val_t), y_val_t).item()
                self.val_loss_history.append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    patience_ctr = 0
                    best_state = {
                        k: v.cpu().clone() for k, v in self._arch.state_dict().items()
                    }
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break
            else:
                if abs(prev_loss - current_loss) < train_tol:
                    break
                prev_loss = current_loss

        if has_val and best_state is not None:
            self._arch.load_state_dict(best_state)
        return self

    def predict(self, X) -> np.ndarray:
        """Run inference; returns numpy array of shape (N, len(out_idx))."""
        if self._arch is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        self._arch.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            return self._arch(X_t).cpu().numpy()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, out_idx={self.out_idx})"


# ---------------------------------------------------------------------------
# Ensemble of per-group GMMs
# ---------------------------------------------------------------------------
class EnsembleGMM:
    """
    Ensemble of SingleGroupGMM estimators, one per output group.

    Each group covers a physically coherent subset of targets (e.g., IMs + Energy,
    Duration parameters, frequency bounds). This reduces negative transfer
    between physically dissimilar outputs.
    """

    def __init__(
        self, input_dim: int, output_dim: int, configs: dict, device: str = "cpu"
    ):
        self.output_dim = output_dim
        self.device = device

        self.models: dict[str, SingleGroupGMM] = {
            grp["name"]: SingleGroupGMM(
                input_dim=input_dim,
                out_idx=grp["out_idx"],
                mono_in=grp["mono_in"],
                signs=grp["signs"],
                config=configs[grp["name"]],
                device=device,
            )
            for grp in group_definitions
        }

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit each group estimator on its slice of the target columns."""
        for grp in group_definitions:
            name = grp["name"]
            idx = grp["out_idx"]
            y_slc = y[:, idx]
            yv_slc = y_val[:, idx] if y_val is not None else None
            self.models[name].fit(X, y_slc, X_val, yv_slc)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict all targets by reassembling slices from each group."""
        preds = np.zeros((X.shape[0], self.output_dim), dtype=np.float32)
        for grp in group_definitions:
            preds[:, grp["out_idx"]] = self.models[grp["name"]].predict(X)
        return preds

    # ── Convenience properties ──────────────────────────────────────────────
    @property
    def train_loss_history(self) -> dict:
        return {name: m.train_loss_history for name, m in self.models.items()}

    @property
    def val_loss_history(self) -> dict:
        return {name: m.val_loss_history for name, m in self.models.items()}

    @property
    def configs(self) -> dict:
        return {name: m.config for name, m in self.models.items()}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"output_dim={self.output_dim}, "
            f"groups={list(self.models)}, "
            f"device='{self.device}')"
        )
