import json
import gzip
import pickle


def read_jsonlist(filename):
    data = []
    with gzip.open(filename, "rt") as fin:
        for line in fin:
            data.append(json.loads(line))
    return data


def write_jsonlist(dataset, filename):
    with gzip.open(filename, "wt") as fout:
        for row in dataset:
            fout.write(json.dumps(row) + "\n")


def load_data(source_names, use_data=True, use_counters=True, data_path="./data"):
    '''
    source_names: list of source names e.g., ["Trump", "Biden"]
    use_data: whether to load the text data
    use_counters: whether to load the corpus word counters
    data_path: path to the data folder
    '''
    data = {}
    corpus_counters ={}

    for src_name in source_names:
        # TODO change
        if use_data:
            data[src_name] = read_jsonlist(f"{data_path}/{src_name.lower()}_public_summaries.jsonl.gz")
            print(f"Loaded {len(data[src_name])} rows")
        if use_counters:
            with open(f"{data_path}/{src_name.lower()}_public_corpus_counter.pkl", "rb") as f:
                corpus_counters[src_name] = pickle.load(f)
                print("loaded counters")

    return data, corpus_counters
