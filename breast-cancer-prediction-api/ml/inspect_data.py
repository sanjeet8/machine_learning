from sklearn.datasets import load_breast_cancer

def main():
    # Load the dataset
    dataset = load_breast_cancer(as_frame=True)

    # Separate input features and target labels
    X = dataset.data
    y = dataset.target

    print("=== Dataset Information ===")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    print("\n=== Feature Names ===")
    print(X.columns.tolist())

    print("\n=== Target Classes ===")
    print(dataset.target_names)
    print("Target distribution:")
    print(y.value_counts().sort_index())

    print("\n=== First 5 Rows ===")
    print(X.head())

    print("\n=== Missing Values ===")
    print(X.isnull().sum().sum())


if __name__ == "__main__":
    main()