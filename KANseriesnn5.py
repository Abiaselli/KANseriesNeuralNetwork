import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

# Simulate a single Term in the additive series
class Term(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.randn(1))
        self.weight = nn.Parameter(torch.randn(1))
        self.route = nn.Parameter(torch.randn(3))  # forward, backward, sideways

    def forward(self, target, neighbor=None):
        route_weights = F.softmax(self.route, dim=0)  # ensure they sum to 1
        fwd = route_weights[0] * self.weight * self.value
        bwd = -route_weights[1] * self.weight * self.value
        side = route_weights[2] * neighbor.weight * neighbor.value if neighbor else 0.0
        return fwd + bwd + side

class SymbolicTerm(nn.Module):
    def __init__(self, func_type="sin"):
        super().__init__()
        self.func_type = func_type
        self.a = nn.Parameter(torch.randn(1))  # scale
        self.b = nn.Parameter(torch.randn(1))  # freq/multiplier
        self.c = nn.Parameter(torch.randn(1))  # phase/offset
        self.weight = nn.Parameter(torch.randn(1))
        self.route = nn.Parameter(torch.randn(3))  # fwd/bwd/side

    def safe_input(self, x):
        return torch.clamp(x, 1e-6, 1e6)  # for log, tan, etc.

    def compute(self, x):
        x = self.safe_input(x)
        t = self.func_type
        if t == "sin":
            return self.a * torch.sin(self.b * x + self.c)
        elif t == "cos":
            return self.a * torch.cos(self.b * x + self.c)
        elif t == "tan":
            return self.a * torch.tan(self.b * x + self.c)
        elif t == "csc":
            return self.a / torch.sin(self.b * x + self.c)
        elif t == "sec":
            return self.a / torch.cos(self.b * x + self.c)
        elif t == "cot":
            return self.a / torch.tan(self.b * x + self.c)
        elif t == "arcsin":
            return self.a * torch.arcsin(torch.clamp(self.b * x + self.c, -1 + 1e-3, 1 - 1e-3))
        elif t == "arccos":
            return self.a * torch.arccos(torch.clamp(self.b * x + self.c, -1 + 1e-3, 1 - 1e-3))
        elif t == "arctan":
            return self.a * torch.arctan(self.b * x + self.c)
        elif t == "exp":
            return self.a * torch.exp(torch.clamp(self.b * x + self.c, max=10))
        elif t == "log":
            return self.a * torch.log(torch.clamp(self.b * x + self.c, min=1e-3))
        elif t == "poly1":
            return self.a * x + self.b
        elif t == "poly2":
            return self.a * x**2 + self.b * x + self.c
        elif t == "poly3":
            return self.a * x**3 + self.b * x**2 + self.c
        else:
            return self.a * x  # fallback

    def forward(self, x, neighbor=None):
        value = self.compute(x)
        route_weights = F.softmax(self.route, dim=0)
        fwd = route_weights[0] * self.weight * value
        bwd = -route_weights[1] * self.weight * value
        side = route_weights[2] * neighbor.compute(x) if neighbor else 0.0
        return fwd + bwd + side


# Controller that generates evolving target values
class GoalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase = 0.0

    def update_goal(self, step):
        # Evolve goal over time (e.g., a sine wave or drifting value)
        self.phase += 0.1
        return torch.tensor([torch.sin(torch.tensor(self.phase))])

# Series that aggregates active terms
class ScalarLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = nn.Linear(1, 1)

    def forward(self, x):
        return self.scalar(x)

class SeriesSimulator(nn.Module):
    def __init__(self, num_terms):
        super().__init__()
        self.scalar_layer = ScalarLayer()
        funcs = [
            "sin", "cos", "tan", "csc", "sec", "cot",
            "arcsin", "arccos", "arctan",
            "exp", "log",
            "poly1", "poly2", "poly3"
        ]
        self.terms = nn.ModuleList([SymbolicTerm(func_type=random.choice(funcs)) for _ in range(num_terms)])
        self.goal_gen = GoalGenerator()

    def forward(self, step):
        x = torch.tensor([[step / 100.0]])
        scalar_out = self.scalar_layer(x)

        # Combine base scalar with symbolic input
        x_symbolic = x + scalar_out

        target = self.goal_gen.update_goal(step)
        total = 0.0
        for i, term in enumerate(self.terms):
            neighbor = self.terms[i + 1] if i + 1 < len(self.terms) else None
            total += term(x_symbolic, neighbor)

        loss = F.smooth_l1_loss(total, target)
        return total, target, loss

def analyze_model(model, step):
    func_counts = {}
    coef_stats = {"a": [], "b": [], "c": []}
    routing_entropy = []

    for term in model.terms:
        func = term.func_type
        func_counts[func] = func_counts.get(func, 0) + 1

        coef_stats["a"].append(term.a.item())
        coef_stats["b"].append(term.b.item())
        coef_stats["c"].append(term.c.item())

        route_probs = F.softmax(term.route, dim=0)
        entropy = -(route_probs * torch.log(route_probs + 1e-8)).sum().item()
        routing_entropy.append(entropy)

    print(f"\n[Diagnostics at step {step}]")
    print("Function distribution:", func_counts)
    print("Mean a/b/c:", {k: round(sum(v) / len(v), 4) for k, v in coef_stats.items()})
    print("Avg routing entropy:", round(sum(routing_entropy) / len(routing_entropy), 4))

# Training loop
def train_series(num_terms=50, steps=1001, lr=0.25):
    print(f"num_terms: {num_terms} steps: {steps} lr={lr}")
    model = SeriesSimulator(num_terms)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


    for step in range(steps):
        optimizer.zero_grad()
        output, target, loss = model(step)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.6f} | Output: {output.item():.4f} | Target: {target.item():.4f}")
        if step % 200 == 0:
            analyze_model(model, step)


    return model

# Run it
model = train_series()
