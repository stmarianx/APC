"""Distill the equal-prior AKQ river solver across bet-to-pot ratios."""
from pathlib import Path
import torch
from torch import nn
from coach.src.poker_coach.river_solver import solve_akq_river


WEIGHTS = Path(__file__).resolve().parents[1] / "checkpoints/river-policy-v1.pt"


def network():
    return nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 3), nn.Sigmoid())


def train(output):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    bets = torch.arange(2, 21, dtype=torch.float32)
    inputs = (bets / 10).log().unsqueeze(1)
    targets = []
    for bet in bets.tolist():
        _, sol = solve_akq_river(pot_bb=10, bet_bb=bet, iterations=40000)
        targets.append([sol.ip_strategy["Value"]["bet"], sol.ip_strategy["Air"]["bet"],
                        sol.oop_strategy["Bluff-catcher"]["call"]])
    targets = torch.tensor(targets)
    held_out = bets.remainder(2) == 1
    model = network()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(1200):
        loss = (model(inputs[~held_out]) - targets[~held_out]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        errors = (model(inputs[held_out]) - targets[held_out]).abs()
    report = {"test_bets_bb": bets[held_out].tolist(), "train_bets_bb": bets[~held_out].tolist(),
              "pot_bb": 10, "seed": 42, "cfr_iterations": 40000,
              "test_frequency_mae": errors.mean().item(), "test_max_error": errors.max().item()}
    torch.save({"weights": model.state_dict(), "evaluation": report}, output)
    return report


def predict(pot_bb, bet_bb):
    if pot_bb <= 0 or not .2 <= bet_bb / pot_bb <= 2:
        raise ValueError("River policy covers bet sizes from 0.2 to 2 times the pot")
    model = network()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True)["weights"])
    with torch.inference_mode():
        return model(torch.tensor([[bet_bb / pot_bb]]).log()).flatten().tolist()


if __name__ == "__main__":
    print(train(WEIGHTS))
