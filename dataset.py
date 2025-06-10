import uproot

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