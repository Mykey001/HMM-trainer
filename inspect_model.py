"""Quick inspection of the existing GMM model and scaler."""
import joblib
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model
gmm = joblib.load(os.path.join(script_dir, "market_regime_gmm.pkl"))
scaler = joblib.load(os.path.join(script_dir, "scaler.pkl"))

print("=" * 60)
print("GMM MODEL INSPECTION")
print("=" * 60)
print(f"Type: {type(gmm).__name__}")
print(f"N components: {gmm.n_components}")
print(f"Covariance type: {gmm.covariance_type}")
print(f"Converged: {gmm.converged_}")
print(f"N iterations: {gmm.n_iter_}")
print(f"\nMeans (shape {gmm.means_.shape}):")
print(gmm.means_)
print(f"\nWeights: {gmm.weights_}")

print("\n" + "=" * 60)
print("SCALER INSPECTION")
print("=" * 60)
print(f"Type: {type(scaler).__name__}")
print(f"Mean: {scaler.mean_}")
print(f"Scale: {scaler.scale_}")
if hasattr(scaler, 'feature_names_in_'):
    print(f"Feature names: {scaler.feature_names_in_}")

# Check CSV data
csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
for csv_file in csv_files:
    csv_path = os.path.join(script_dir, csv_file)
    print(f"\n{'=' * 60}")
    print(f"CSV DATA: {csv_file}")
    print(f"{'=' * 60}")
    df = pd.read_csv(csv_path, nrows=5)
    print(f"Columns: {list(df.columns)}")
    print(f"Shape (first 5 rows): {df.shape}")
    print(df.head())
    # Get total row count
    total = sum(1 for _ in open(csv_path)) - 1
    print(f"\nTotal rows: {total:,}")

print("\n\nDONE - Model inspection complete.")
