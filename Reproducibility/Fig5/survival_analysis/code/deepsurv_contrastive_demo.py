"""
UniCure survival analysis workflow — Fig. 5L-O and Fig. S16I-K

This script documents the survival model training logic used to associate
UniCure-ranked administered therapies with patient outcomes. The Kaplan-Meier
plotting workflow uses released patient-level survival and drug-rank tables from
./raw_data/fig5/survival_analysis/.

Model: UniCureSurvivalHead — a risk head on top of the pretrained UniCure model.
Pretrained weights: \path_to\Unicure_best_model.pth

Training jointly optimizes:
  1. Cox partial likelihood loss on patient-level administered-drug risk scores.
  2. Contrastive loss constraining administered-drug risks below random-drug risks.

After training, patient-level drug-rank summaries can be grouped using the
manuscript survival rule and visualized with Kaplan-Meier curves.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from utils import load_UniCure_pretrained_model


class UniCureSurvivalHead(nn.Module):
    def __init__(self, unicure_model, expression_dim=978, clinical_dim=3, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.unicure = unicure_model
        self.risk_head = nn.Sequential(
            nn.Linear(expression_dim + clinical_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, cell_emb, drug_emb, clinical_covariates):
        predicted_expression = self.unicure("pertrub_forward", cell_emb, drug_emb)
        risk_input = torch.cat([predicted_expression, clinical_covariates], dim=1)
        risk_score = self.risk_head(risk_input).squeeze(1)
        return risk_score, predicted_expression

    def forward_patient_drug_sets(self, cell_emb, administered_drug_emb, random_drug_emb, clinical_covariates):
        batch_size, n_administered, drug_dim = administered_drug_emb.shape
        n_random = random_drug_emb.shape[1]

        administered_cell = cell_emb.unsqueeze(1).expand(-1, n_administered, -1).reshape(batch_size * n_administered, -1)
        administered_clinical = clinical_covariates.unsqueeze(1).expand(-1, n_administered, -1).reshape(batch_size * n_administered, -1)
        administered_drug = administered_drug_emb.reshape(batch_size * n_administered, drug_dim)

        random_cell = cell_emb.unsqueeze(1).expand(-1, n_random, -1).reshape(batch_size * n_random, -1)
        random_clinical = clinical_covariates.unsqueeze(1).expand(-1, n_random, -1).reshape(batch_size * n_random, -1)
        random_drug = random_drug_emb.reshape(batch_size * n_random, drug_dim)

        administered_pair_risk, administered_pair_expr = self.forward(administered_cell, administered_drug, administered_clinical)
        random_pair_risk, random_pair_expr = self.forward(random_cell, random_drug, random_clinical)

        administered_pair_risk = administered_pair_risk.reshape(batch_size, n_administered)
        random_pair_risk = random_pair_risk.reshape(batch_size, n_random)
        administered_pair_expr = administered_pair_expr.reshape(batch_size, n_administered, -1)
        random_pair_expr = random_pair_expr.reshape(batch_size, n_random, -1)

        patient_administered_risk = administered_pair_risk.mean(dim=1)
        patient_random_risk = random_pair_risk.mean(dim=1)

        return {
            "patient_administered_risk": patient_administered_risk,
            "patient_random_risk": patient_random_risk,
            "administered_pair_risk": administered_pair_risk,
            "random_pair_risk": random_pair_risk,
            "administered_pair_expr": administered_pair_expr,
            "random_pair_expr": random_pair_expr,
        }


def cox_partial_likelihood_loss(risk_score, survival_time, event_observed):
    order = torch.argsort(survival_time, descending=True)
    risk_score = risk_score[order]
    event_observed = event_observed[order]
    log_risk_set_sum = torch.logcumsumexp(risk_score, dim=0)
    observed = event_observed > 0
    return -torch.mean(risk_score[observed] - log_risk_set_sum[observed])


def administered_vs_random_contrastive_loss(administered_risk, random_risk, margin=0.1):
    pairwise_gap = margin + administered_risk.unsqueeze(2) - random_risk.unsqueeze(1)
    return F.relu(pairwise_gap).mean()


def training_step(model, batch, optimizer, contrastive_weight=0.5, margin=0.1):
    cell_emb = batch["cell_emb"]
    administered_drug_emb = batch["administered_drug_emb_set"]
    random_drug_emb = batch["random_drug_emb_set"]
    clinical_covariates = batch["clinical_covariates"]
    survival_time = batch["survival_time"]
    event_observed = batch["event_observed"]

    outputs = model.forward_patient_drug_sets(cell_emb, administered_drug_emb, random_drug_emb, clinical_covariates)
    patient_administered_risk = outputs["patient_administered_risk"]
    patient_random_risk = outputs["patient_random_risk"]
    administered_pair_risk = outputs["administered_pair_risk"]
    random_pair_risk = outputs["random_pair_risk"]

    cox_loss = cox_partial_likelihood_loss(patient_administered_risk, survival_time, event_observed)
    contrastive_loss = administered_vs_random_contrastive_loss(administered_pair_risk, random_pair_risk, margin)
    total_loss = cox_loss + contrastive_weight * contrastive_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "loss": total_loss.detach(),
        "cox_loss": cox_loss.detach(),
        "contrastive_loss": contrastive_loss.detach(),
        "patient_administered_risk": patient_administered_risk.detach(),
        "patient_random_risk": patient_random_risk.detach(),
        "administered_pair_risk": outputs["administered_pair_risk"].detach(),
        "random_pair_risk": outputs["random_pair_risk"].detach(),
    }


def train_survival_model(train_loader, model_path, device="cuda:0", epochs=50, lr=1e-4):
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    unicure = load_UniCure_pretrained_model(path=model_path)
    model = UniCureSurvivalHead(unicure).to(device)

    for param in model.unicure.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.risk_head.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            metrics = training_step(model, batch, optimizer)
        print(
            f"epoch={epoch + 1} "
            f"loss={metrics['loss'].item():.4f} "
            f"cox={metrics['cox_loss'].item():.4f} "
            f"contrast={metrics['contrastive_loss'].item():.4f}"
        )

    return model


def expected_batch_format():
    return {
        "cell_emb": "FloatTensor[batch, 1280], patient UCE/UniCure cell embedding",
        "administered_drug_emb_set": "FloatTensor[batch, n_administered, 528], all matched true administered therapy inputs",
        "random_drug_emb_set": "FloatTensor[batch, n_random, 528], randomly sampled non-administered therapy inputs",
        "clinical_covariates": "FloatTensor[batch, 3], for example age, gender, stage",
        "survival_time": "FloatTensor[batch], overall survival or follow-up time",
        "event_observed": "FloatTensor[batch], 1=death/event, 0=right-censored",
    }


if __name__ == "__main__":
    model_path = Path(r"\path_to\Unicure_best_model.pth")
    print("This file is a logic demo. Build a DataLoader that yields:")
    for key, value in expected_batch_format().items():
        print(f"  {key}: {value}")
    print(f"Then call train_survival_model(train_loader, model_path={str(model_path)!r}).")
