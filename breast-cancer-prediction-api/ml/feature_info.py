
from sklearn.datasets import load_breast_cancer


def main():
    dataset = load_breast_cancer()

    print(f"Total features: {len(dataset.feature_names)}")
    print("\nFeature names in model input order:\n")

    for index, name in enumerate(dataset.feature_names):
        print(f"{index + 1:2}. {name}")


if __name__ == "__main__":
    main()
