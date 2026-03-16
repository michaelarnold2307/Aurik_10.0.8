"""
Advanced Ensemble Strategies für Aurik 8.0

Implementiert fortgeschrittene Ensemble-Methoden:
- Stacking mit Meta-Learner
- Meta-Learning für adaptive Kombinationen
- Weighted Voting mit gelernten Gewichten
- Dynamic Ensemble Selection
- Mixture of Experts

Autor: Aurik Backend-Team
Version: 8.2
Datum: 14. Februar 2026
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class EnsembleMember:
    """Data class for ensemble member information."""

    name: str
    model: nn.Module
    weight: float = 1.0
    performance: float = 0.0
    specialization: str | None = None  # e.g., "denoise", "declip", "declick"


class MetaLearner(nn.Module):
    """
    Meta-Learner für Stacking Ensemble.

    Lernt, wie die Vorhersagen der Base-Models optimal kombiniert werden.
    """

    def __init__(self, n_base_models: int, feature_dim: int = 64, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__()

        self.n_base_models = n_base_models

        # Feature extractor for each base model output
        self.feature_extractors = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(output_dim, feature_dim), nn.ReLU(), nn.Dropout(0.2))
                for _ in range(n_base_models)
            ]
        )

        # Meta-learner network
        self.meta_network = nn.Sequential(
            nn.Linear(n_base_models * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        logger.info(f"MetaLearner initialized: {n_base_models} base models, {feature_dim} features")

    def forward(self, base_predictions: list[torch.Tensor]) -> torch.Tensor:
        """
        Combine base model predictions using meta-learning.

        Args:
            base_predictions: List of predictions from base models [batch_size, output_dim]

        Returns:
            Combined prediction [batch_size, output_dim]
        """
        # Extract features from each base prediction
        features = []
        for pred, extractor in zip(base_predictions, self.feature_extractors):
            features.append(extractor(pred))

        # Concatenate all features
        combined_features = torch.cat(features, dim=1)

        # Meta-learner prediction
        output = self.meta_network(combined_features)

        return output


class AttentionWeightPredictor(nn.Module):
    """
    Lernt dynamische Gewichte für Ensemble-Members basierend auf Input-Features.

    Verwendet Attention-Mechanismus zur adaptiven Gewichtung.
    """

    def __init__(self, n_members: int, input_feature_dim: int = 128, attention_dim: int = 64):
        super().__init__()

        self.n_members = n_members

        # Query, Key, Value projections
        self.query = nn.Linear(input_feature_dim, attention_dim)
        self.keys = nn.ModuleList([nn.Linear(input_feature_dim, attention_dim) for _ in range(n_members)])
        self.values = nn.ModuleList([nn.Linear(input_feature_dim, attention_dim) for _ in range(n_members)])

        # Output projection
        self.output_proj = nn.Linear(attention_dim, 1)

        logger.info(f"AttentionWeightPredictor initialized: {n_members} members")

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Predict ensemble weights based on input features.

        Args:
            input_features: Input features [batch_size, feature_dim]

        Returns:
            Weights for each ensemble member [batch_size, n_members]
        """
        input_features.size(0)

        # Compute query
        q = self.query(input_features)  # [batch, attention_dim]

        # Compute attention scores for each member
        scores = []
        for key_proj in self.keys:
            k = key_proj(input_features)  # [batch, attention_dim]
            score = torch.sum(q * k, dim=1, keepdim=True)  # [batch, 1]
            scores.append(score)

        scores = torch.cat(scores, dim=1)  # [batch, n_members]

        # Softmax to get weights
        weights = F.softmax(scores, dim=1)

        return weights


class DynamicEnsembleSelector(nn.Module):
    """
    Dynamic Ensemble Selection basierend auf Input-Charakteristiken.

    Wählt für jeden Input dynamisch die besten k Ensemble-Members aus.
    """

    def __init__(self, n_members: int, feature_extractor: nn.Module, k_select: int = 3):
        super().__init__()

        self.n_members = n_members
        self.k_select = min(k_select, n_members)
        self.feature_extractor = feature_extractor

        # Selector network
        self.selector = nn.Sequential(
            nn.Linear(128, 64),  # Assuming feature_extractor outputs 128-dim
            nn.ReLU(),
            nn.Linear(64, n_members),
            nn.Sigmoid(),  # Selection probabilities
        )

        logger.info(f"DynamicEnsembleSelector initialized: top-{k_select} from {n_members}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select ensemble members dynamically.

        Args:
            x: Input tensor

        Returns:
            selection_mask: Binary mask [batch, n_members]
            selection_scores: Selection probabilities [batch, n_members]
        """
        # Extract features
        features = self.feature_extractor(x)

        # Predict selection probabilities
        scores = self.selector(features)

        # Select top-k
        top_k_values, top_k_indices = torch.topk(scores, self.k_select, dim=1)

        # Create selection mask
        mask = torch.zeros_like(scores)
        mask.scatter_(1, top_k_indices, 1.0)

        return mask, scores


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts Ensemble.

    Jeder Expert spezialisiert sich auf verschiedene Input-Regionen.
    Gating Network entscheidet, welche Experts aktiviert werden.
    """

    def __init__(self, experts: list[nn.Module], input_dim: int, k_active: int = 2) -> None:
        super().__init__()

        self.experts = nn.ModuleList(experts)
        self.n_experts = len(experts)
        self.k_active = k_active

        # Gating network
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_experts),
        )

        # Load balancing loss weight
        self.load_balance_weight = 0.01

        logger.info(f"MixtureOfExperts initialized: {self.n_experts} experts, top-{k_active} active")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass through mixture of experts.

        Args:
            x: Input tensor

        Returns:
            output: Combined expert outputs
            aux_loss_dict: Auxiliary losses (e.g., load balancing)
        """
        x.size(0)

        # Compute gating logits
        gate_logits = self.gating_network(x)  # [batch, n_experts]

        # Top-k gating with sparsity
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.k_active, dim=1)
        top_k_gates = F.softmax(top_k_logits, dim=1)

        # Create sparse gate tensor
        gates = torch.zeros_like(gate_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)

        # Expert predictions
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))

        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, n_experts, output_dim]

        # Weighted combination
        gates_expanded = gates.unsqueeze(-1)  # [batch, n_experts, 1]
        output = torch.sum(gates_expanded * expert_outputs, dim=1)

        # Load balancing loss (encourage uniform expert usage)
        importance = gates.sum(dim=0)  # [n_experts]
        load_balance_loss = self.load_balance_weight * torch.var(importance)

        aux_loss_dict = {"load_balance_loss": load_balance_loss, "expert_importance": importance}

        return output, aux_loss_dict


class AdvancedEnsemble:
    """
    Main Advanced Ensemble class that combines all strategies.

    Unterstützt:
    - Stacking with Meta-Learner
    - Dynamic Weighted Voting
    - Ensemble Selection
    - Mixture of Experts
    """

    def __init__(
        self,
        ensemble_members: list[EnsembleMember],
        strategy: str = "stacking",
        device: str = "cpu",  # §9.5: Aurik 9 — ausschließlich CPU, kein CUDA/ROCm/Metal
    ) -> None:
        """
        Initialize advanced ensemble.

        Args:
            ensemble_members: List of ensemble members
            strategy: "stacking", "weighted_voting", "dynamic_selection", "mixture_of_experts"
            device: Device to use
        """
        self.members = ensemble_members
        self.strategy = strategy
        self.device = device

        # Move all models to device
        for member in self.members:
            member.model = member.model.to(device)

        # Initialize strategy-specific components
        if strategy == "stacking":
            self.meta_learner = MetaLearner(
                n_base_models=len(ensemble_members), feature_dim=64, hidden_dim=128, output_dim=1
            ).to(device)

        elif strategy == "weighted_voting":
            # Simple feature extractor for attention weights
            feature_extractor = nn.Sequential(
                nn.Conv1d(1, 32, 15, padding=7),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(32, 64, 7, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            ).to(device)

            self.weight_predictor = AttentionWeightPredictor(n_members=len(ensemble_members), input_feature_dim=64).to(
                device
            )
            self.feature_extractor = feature_extractor

        elif strategy == "dynamic_selection":
            feature_extractor = nn.Sequential(
                nn.Conv1d(1, 32, 15, padding=7),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(32, 64, 7, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, 128),
                nn.ReLU(),
            ).to(device)

            self.selector = DynamicEnsembleSelector(
                n_members=len(ensemble_members),
                feature_extractor=feature_extractor,
                k_select=max(2, len(ensemble_members) // 2),
            ).to(device)

        elif strategy == "mixture_of_experts":
            # Assume members are experts
            self.moe = MixtureOfExperts(
                experts=[m.model for m in ensemble_members],
                input_dim=48000,  # Assuming 1 second at 48kHz
                k_active=max(2, len(ensemble_members) // 3),
            ).to(device)

        logger.info(f"AdvancedEnsemble initialized: strategy={strategy}, {len(ensemble_members)} members")

    def predict(self, x: torch.Tensor, return_details: bool = False) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """
        Make ensemble prediction.

        Args:
            x: Input tensor [batch, channels, samples]
            return_details: Whether to return detailed information

        Returns:
            prediction: Ensemble prediction
            details: Optional details dict
        """
        x = x.to(self.device)
        details = {} if return_details else None

        if self.strategy == "stacking":
            # Get predictions from all base models
            base_predictions = []
            for member in self.members:
                with torch.no_grad():
                    pred = member.model(x)
                base_predictions.append(pred)

            # Meta-learner combines predictions
            prediction = self.meta_learner(base_predictions)

            if return_details:
                details["base_predictions"] = base_predictions
                details["n_members"] = len(self.members)

        elif self.strategy == "weighted_voting":
            # Extract features for weight prediction
            features = self.feature_extractor(x)
            weights = self.weight_predictor(features)

            # Get weighted predictions
            predictions = []
            for member in self.members:
                with torch.no_grad():
                    pred = member.model(x)
                predictions.append(pred)

            predictions = torch.stack(predictions, dim=1)  # [batch, n_members, ...]
            # Expand weights to match prediction dimensions
            while weights.dim() < predictions.dim():
                weights = weights.unsqueeze(-1)

            prediction = torch.sum(weights * predictions, dim=1)

            if return_details:
                details["weights"] = weights
                details["predictions"] = predictions

        elif self.strategy == "dynamic_selection":
            # Select best members for this input
            selection_mask, selection_scores = self.selector(x)

            # Get predictions from selected members
            predictions = []
            for member in self.members:
                with torch.no_grad():
                    pred = member.model(x)
                predictions.append(pred)

            predictions = torch.stack(predictions, dim=1)
            mask_expanded = selection_mask.unsqueeze(-1)

            # Average selected predictions
            selected_sum = torch.sum(mask_expanded * predictions, dim=1)
            selected_count = selection_mask.sum(dim=1, keepdim=True)
            prediction = selected_sum / (selected_count + 1e-8)

            if return_details:
                details["selection_mask"] = selection_mask
                details["selection_scores"] = selection_scores

        elif self.strategy == "mixture_of_experts":
            # MoE forward
            prediction, aux_losses = self.moe(x)

            if return_details:
                details["aux_losses"] = aux_losses

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return prediction, details

    def train_meta_learner(self, train_loader, val_loader, epochs: int = 20, lr: float = 1e-3) -> None:
        """
        Train meta-learner for stacking strategy.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
        """
        if self.strategy != "stacking":
            logger.warning(f"Meta-learner training only for stacking strategy, current: {self.strategy}")
            return

        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=lr)

        logger.info(f"Training meta-learner for {epochs} epochs...")

        for epoch in range(epochs):
            # Training
            self.meta_learner.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Get base predictions (no gradients)
                base_predictions = []
                for member in self.members:
                    with torch.no_grad():
                        pred = member.model(batch_x)
                    base_predictions.append(pred)

                # Meta-learner forward
                optimizer.zero_grad()
                meta_pred = self.meta_learner(base_predictions)
                loss = F.mse_loss(meta_pred, batch_y)

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            self.meta_learner.eval()
            val_losses = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    base_predictions = []
                    for member in self.members:
                        pred = member.model(batch_x)
                        base_predictions.append(pred)

                    meta_pred = self.meta_learner(base_predictions)
                    loss = F.mse_loss(meta_pred, batch_y)

                    val_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        logger.info("Meta-learner training completed!")

    def save(self, path: Path) -> None:
        """Save ensemble configuration and learned components."""
        save_dict = {
            "strategy": self.strategy,
            "n_members": len(self.members),
            "member_info": [
                {"name": m.name, "weight": m.weight, "performance": m.performance, "specialization": m.specialization}
                for m in self.members
            ],
        }

        # Save strategy-specific components
        if self.strategy == "stacking":
            torch.save(self.meta_learner.state_dict(), path.parent / f"{path.stem}_meta_learner.pth")
        elif self.strategy == "weighted_voting":
            torch.save(self.weight_predictor.state_dict(), path.parent / f"{path.stem}_weight_predictor.pth")
            torch.save(self.feature_extractor.state_dict(), path.parent / f"{path.stem}_feature_extractor.pth")
        elif self.strategy == "dynamic_selection":
            torch.save(self.selector.state_dict(), path.parent / f"{path.stem}_selector.pth")
        elif self.strategy == "mixture_of_experts":
            torch.save(self.moe.state_dict(), path.parent / f"{path.stem}_moe.pth")

        with open(path, "w") as f:
            json.dump(save_dict, f, indent=2)

        logger.info(f"Ensemble saved to {path}")


# Example usage
if __name__ == "__main__":
    # Create dummy ensemble members
    members = []
    for i in range(3):
        model = nn.Sequential(nn.Conv1d(1, 32, 15, padding=7), nn.ReLU(), nn.Conv1d(32, 1, 1))
        members.append(EnsembleMember(name=f"model_{i}", model=model, weight=1.0))

    # Test each strategy
    for strategy in ["stacking", "weighted_voting", "dynamic_selection"]:
        logger.debug(f"\n=== Testing {strategy} ===")
        ensemble = AdvancedEnsemble(members, strategy=strategy, device="cpu")

        x = torch.randn(2, 1, 48000)
        prediction, details = ensemble.predict(x, return_details=True)

        logger.debug(f"Prediction shape: {prediction.shape}")
        if details:
            logger.debug(f"Details keys: {list(details.keys())}")
