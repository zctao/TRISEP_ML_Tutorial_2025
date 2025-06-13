import uproot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset(filename, treename="tree_event"):
    """
    Load dataset from a ROOT file into a pandas DataFrame.
    """
    with uproot.open(filename) as file:
        tree = file[treename]
        # Load events as pandas DataFrame
        dataset = tree.arrays(library="pd")
        print(f"Loaded {len(dataset)} events from {filename}.")

    # Shuffle data
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # Examine dataset
    # Print first few events
    #print(dataset.head())

    # Print dataset statistics
    #print(dataset.describe())

    label_weights = ( dataset[dataset.label==0].mcWeight.sum(), dataset[dataset.label==1].mcWeight.sum() )
    print(f"Sum of label weights: Background, Signal = {label_weights}")

    label_nevents = ( dataset[dataset.label==0].shape[0], dataset[dataset.label==1].shape[0] )
    print(f"Total class number of events: Background, Signal = {label_nevents}")

    # Event selection
    # Select events with exactly 2 leptons and positive weights
    print(f"Dataset shape before selection: {dataset.shape}")
    dataset = dataset[(dataset.lep_n == 2) & (dataset.mcWeight > 0)]
    print(f"Dataset shape after selection: {dataset.shape}")

    return dataset

def preprocess_dataset(features_df, target, weights, test_size=0.25):
    """
    Preprocess the dataset by splitting it into training, validation, and test sets.
    """
    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        features_df, target, weights, test_size=test_size, shuffle=True
    )

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    w_train = w_train.reset_index(drop=True)
    w_test = w_test.reset_index(drop=True)

    # Further split test set into validation and test sets
    X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
        X_test, y_test, w_test, test_size=0.5, shuffle=True
    )

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, w_train shape: {w_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}, w_val shape: {w_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, w_test shape: {w_test.shape}")

    # Standardize the features
    # scale to mean=0, std=1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Adjust event weights: train on equal amount of signal and background events; test with original weights
    class_weights_train = (w_train[y_train == 0].sum(), w_train[y_train == 1].sum())
    print(f"class_weights_train: Background, Signal = {class_weights_train}")
    for i in range(len(class_weights_train)):
        w_train[y_train == i] *= max(class_weights_train) / class_weights_train[i]  # equalize number of background and signal events
        w_test[y_test == i] *= 1 / test_size / 0.5  # increase test weight to compensate for sampling

    print(f"Train weights: Background, Signal = {w_train[y_train == 0].sum()}, {w_train[y_train == 1].sum()}")
    print(f"Validation weights: Background, Signal = {w_val[y_val == 0].sum()}, {w_val[y_val == 1].sum()}")
    print(f"Test weights: Background, Signal = {w_test[y_test == 0].sum()}, {w_test[y_test == 1].sum()}")

    return X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test