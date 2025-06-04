"""
genetic_feature_selection.py

This script runs a Genetic Algorithm (GA) for feature selection on the Alzheimer’s dataset
under various parameter settings. For each combination of (population_size, crossover_prob,
mutation_prob), it runs the GA 10 times, records the best fitness and number of generations
to termination, and computes average results. It also plots the average evolution curve
(best fitness vs. generation) for each case.

Requirements:
    - Python 3.7+
    - pandas
    - numpy
    - scikit-learn
    - torch (PyTorch)
    - matplotlib
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ──────────── 1. Define the ANN architecture ──────────── #
class DeepANN(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        super(DeepANN, self).__init__()
        layers = []
        prev_dim = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.ReLU())
            prev_dim = size
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ──────────── 2. Custom Dataset for preprocessing ──────────── #
class MedicalDataset(Dataset):
    def __init__(self, df, scaler=None):
        # Extract labels
        self.labels = df['Diagnosis'].values.astype('float32').reshape(-1, 1)
        # Drop unused columns
        df = df.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'])
        # One‐hot encode categorical columns
        categorical_cols = ['Gender', 'Ethnicity', 'EducationLevel']
        df = pd.get_dummies(df, columns=categorical_cols)
        # Identify feature columns
        feature_cols = df.columns
        # Fit or apply scaler
        if scaler is None:
            self.scaler = MinMaxScaler()
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            self.scaler = scaler
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        self.features = df.values.astype('float32')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.labels[idx])
        return x, y

# ──────────── 3. Load and split data ──────────── #
def load_data(csv_path, test_size=0.2, random_state=42):
    """
    Loads the Alzheimer’s CSV, splits into train/validation, and returns DataLoaders.
    """
    df = pd.read_csv(csv_path)
    df.fillna(0, inplace=True)
    # Stratified split
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df['Diagnosis'], random_state=random_state
    )
    train_dataset = MedicalDataset(train_df)
    val_dataset = MedicalDataset(val_df, scaler=train_dataset.scaler)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    input_dim = train_dataset.features.shape[1]
    return train_loader, val_loader, input_dim

# ──────────── 4. Train the base model on all features ──────────── #
def train_base_model(train_loader, input_dim, device,
                     layer_sizes=[78, 39, 20], lr=0.1, momentum=0.6, 
                     epochs=50):
    """
    Train a DeepANN on all features once and return the trained model.
    """
    model = DeepANN(input_dim, layer_sizes).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    best_state = None
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model

# ──────────── 5. Fitness evaluation ──────────── #
def evaluate_individual(model, chromosome, val_loader, device, alpha=0.1):
    """
    Compute penalized BCE loss on val_loader by zeroing out features not in 'chromosome'.
      - chromosome: 1D numpy array (0/1) of length=input_dim.
      - alpha: penalty coefficient.
    Returns: fitness (float).
    """
    mask = torch.tensor(chromosome, dtype=torch.float32, device=device).unsqueeze(0)
    criterion = nn.BCELoss(reduction='mean')
    model.eval()

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            masked = inputs * mask              # zero out dropped features
            outputs = model(masked)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

    avg_loss = total_loss / count
    num_features = int(chromosome.sum())
    penalty = alpha * (num_features / len(chromosome)) * avg_loss
    fitness = avg_loss + penalty
    return fitness

# ──────────── 6. GA operators ──────────── #
def tournament_selection(population, fitnesses, k=3):
    """
    Choose one individual by k‐tournament (lower fitness is better).
    """
    idxs = np.random.choice(len(population), k, replace=False)
    best = idxs[0]
    for idx in idxs[1:]:
        if fitnesses[idx] < fitnesses[best]:
            best = idx
    return population[best]

def two_point_crossover(parent1, parent2):
    """
    Two‐point crossover on 1D numpy arrays of equal length.
    """
    length = len(parent1)
    c1, c2 = sorted(random.sample(range(1, length), 2))
    child1 = parent1.copy()
    child2 = parent2.copy()
    child1[c1:c2] = parent2[c1:c2]
    child2[c1:c2] = parent1[c1:c2]
    return child1, child2

def mutate(chromosome, mutation_rate=0.01):
    """
    Flip each bit with probability=mutation_rate.
    If result is all zeros, re-enable one random bit.
    """
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    if chromosome.sum() == 0:
        idx = random.randrange(len(chromosome))
        chromosome[idx] = 1
    return chromosome

# ──────────── 7. Single GA run with early stopping ──────────── #
def run_ga_once(model, val_loader, device, input_dim,
                pop_size=100, crossover_rate=0.8, mutation_rate=0.01,
                tournament_k=3, elitism_size=2,
                alpha=0.1, max_generations=1000,
                patience=20, improvement_threshold=0.01):
    """
    Perform a single run of the GA with early stopping criteria:
      - Stop if no relative improvement ≥ improvement_threshold (1%) in the best fitness
        over 'patience' consecutive generations, OR
      - generation count reaches max_generations.

    Returns:
      - best_chromosome: 1D numpy array of length=input_dim.
      - best_fitness: float (fitness of best_chromosome).
      - history: list of best_fitness per generation.
      - generations_used: int (number of generations until termination).
    """
    # 1. Initialize population (unique, non-empty)
    population = []
    while len(population) < pop_size:
        chrom = np.random.choice([0, 1], size=(input_dim,))
        if chrom.sum() == 0:
            chrom[random.randrange(input_dim)] = 1
        # ensure uniqueness
        if not any(np.array_equal(chrom, p) for p in population):
            population.append(chrom)

    best_overall = None
    best_overall_fitness = float('inf')
    history = []

    no_improve_count = 0
    prev_best = None

    for gen in range(1, max_generations + 1):
        # Evaluate population
        fitnesses = np.zeros(pop_size, dtype=float)
        for i, chrom in enumerate(population):
            fitnesses[i] = evaluate_individual(model, chrom, val_loader, device, alpha)

        # Track generation‐best
        gen_best_idx = int(np.argmin(fitnesses))
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_best_chrom = population[gen_best_idx].copy()

        history.append(gen_best_fitness)

        # Update overall best
        if gen_best_fitness < best_overall_fitness:
            best_overall_fitness = gen_best_fitness
            best_overall = gen_best_chrom.copy()

        # Check early stopping
        if prev_best is not None:
            rel_improve = (prev_best - gen_best_fitness) / prev_best
            if rel_improve >= improvement_threshold:
                no_improve_count = 0
            else:
                no_improve_count += 1
        prev_best = gen_best_fitness

        if no_improve_count >= patience:
            # No significant improvement in 'patience' generations
            generations_used = gen
            break

        # Elitism: carry top 'elitism_size' forward
        elite_idxs = np.argsort(fitnesses)[:elitism_size]
        new_pop = [population[i].copy() for i in elite_idxs]

        # Generate rest of new population
        while len(new_pop) < pop_size:
            # Selection
            p1 = tournament_selection(population, fitnesses, tournament_k)
            p2 = tournament_selection(population, fitnesses, tournament_k)

            # Crossover
            if random.random() < crossover_rate:
                c1, c2 = two_point_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

    else:
        # If loop completes without break, we hit max_generations
        generations_used = max_generations

    return best_overall, best_overall_fitness, history, generations_used

# ──────────── 8. Experiment loop over parameter cases ──────────── #
def run_experiments(csv_path, output_dir="ga_results"):
    """
    Runs the GA for each parameter combination 10 times, collects averages,
    and plots the average evolution curve. Saves plots and prints a summary table.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 8.1 Parameter cases from the table
    # Each dict: { 'pop_size': int, 'cx_prob': float, 'mut_prob': float }
    param_cases = [
        {"pop_size": 20,  "cx_prob": 0.6, "mut_prob": 0.00},
        {"pop_size": 20,  "cx_prob": 0.6, "mut_prob": 0.01},
        {"pop_size": 20,  "cx_prob": 0.6, "mut_prob": 0.10},
        {"pop_size": 20,  "cx_prob": 0.9, "mut_prob": 0.01},
        {"pop_size": 20,  "cx_prob": 0.1, "mut_prob": 0.01},
        {"pop_size": 200, "cx_prob": 0.6, "mut_prob": 0.00},
        {"pop_size": 200, "cx_prob": 0.6, "mut_prob": 0.01},
        {"pop_size": 200, "cx_prob": 0.6, "mut_prob": 0.10},
        {"pop_size": 200, "cx_prob": 0.9, "mut_prob": 0.01},
        {"pop_size": 200, "cx_prob": 0.1, "mut_prob": 0.01},
    ]

    # 8.2 Load data and train base model once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, input_dim = load_data(csv_path)
    print(f"Feature‐vector dimension after one‐hot: {input_dim}")

    print("Training base ANN on all features...")
    base_model = train_base_model(train_loader, input_dim, device)
    base_model.to(device)
    base_model.eval()

    # 8.3 Prepare summary storage
    summary_rows = []
    case_idx = 1

    # 8.4 Loop over each parameter case
    for case in param_cases:
        pop_size = case["pop_size"]
        cx_prob  = case["cx_prob"]
        mut_prob = case["mut_prob"]

        print(f"\n=== Case {case_idx}: pop={pop_size}, cx={cx_prob}, mut={mut_prob} ===")
        all_best_fitness = []
        all_gen_counts = []
        all_histories = []

        # Run GA 10 times
        for run_i in range(1, 11):
            print(f"  Run {run_i}/10 ...", end=" ", flush=True)
            best_chrom, best_fit, history, gens_used = run_ga_once(
                base_model, val_loader, device, input_dim,
                pop_size=pop_size,
                crossover_rate=cx_prob,
                mutation_rate=mut_prob,
                tournament_k=3,
                elitism_size=2,
                alpha=0.1,
                max_generations=1000,
                patience=20,
                improvement_threshold=0.01
            )
            print(f"Done. BestFit={best_fit:.4f}, Gen={gens_used}")
            all_best_fitness.append(best_fit)
            all_gen_counts.append(gens_used)
            all_histories.append(history)

        # Compute averages
        avg_best_fit = float(np.mean(all_best_fitness))
        avg_gen_count = float(np.mean(all_gen_counts))

        # Prepare average evolution curve
        #  - First determine longest run length
        max_len = max(len(hist) for hist in all_histories)
        #  - Pad each history by repeating last value if shorter
        padded = []
        for hist in all_histories:
            if len(hist) < max_len:
                last_val = hist[-1]
                padded.append(hist + [last_val] * (max_len - len(hist)))
            else:
                padded.append(hist)
        avg_evolution = np.mean(np.vstack(padded), axis=0)

        # 8.5 Save/plot average evolution curve
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_len + 1), avg_evolution, marker='o', markersize=4)
        plt.title(
            f"Avg. Evolution Curve\npop={pop_size}, cx={cx_prob}, mut={mut_prob}"
        )
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness (lower is better)")
        plt.grid(True)
        plt.tight_layout()
        plot_fname = os.path.join(
            output_dir,
            f"evolution_pop{pop_size}_cx{cx_prob}_mut{mut_prob}.png"
        )
        plt.savefig(plot_fname)
        plt.close()
        print(f"  → Saved evolution plot to {plot_fname}")

        # 8.6 Append to summary
        summary_rows.append({
            "Case": case_idx,
            "PopSize": pop_size,
            "CrossoverProb": cx_prob,
            "MutationProb": mut_prob,
            "AvgBestFitness": avg_best_fit,
            "AvgGenerations": avg_gen_count
        })

        case_idx += 1

    # 8.7 Print summary table
    summary_df = pd.DataFrame(summary_rows)
    print("\n=== SUMMARY TABLE ===")
    print(summary_df.to_string(index=False))

    # 8.8 Optionally save summary to CSV
    csv_out = os.path.join(output_dir, "ga_summary.csv")
    summary_df.to_csv(csv_out, index=False)
    print(f"\nSummary table saved to: {csv_out}")


if __name__ == "__main__":
    # Path to the Alzheimer’s dataset CSV
    DATA_PATH = "alzheimers_disease_data.csv"
    run_experiments(DATA_PATH, output_dir="ga_results")
