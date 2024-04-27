# entity-based-political-bias
Data and code for the EMNLP 2023 Findings paper ["Entity-Based Evaluation of Political Bias in Automatic Summarization"](https://aclanthology.org/2023.findings-emnlp.696/) by Karen Zhou and Chenhao Tan. 

[[ACL anthology vers](https://aclanthology.org/2023.findings-emnlp.696/)]
[[arXiv vers](https://arxiv.org/abs/2305.02321)]

## Data

Download here: [Harvard Dataverse link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FSHEVDA)

We release our generated original and replaced summaries for the models [PreSumm, PEGASUS, BART, ProphetNet].

Note that we cannot release full article texts; see [https://www.english-corpora.org/now/](https://www.english-corpora.org/now/) to obtain access. The `textID` column in `data` corresponds to the NOW corpus textIDs, and the url of the article is shared.



For each $e_1 \in$ [Trump, Biden], there are two data files:
- `data` (e.g., "{e_1}_public_summaries.jsonl.gz")
    - a list of dictionaries, where each element corresponds to an article and its corresponding summary versions
    - each row contains the following fields:
        - `textID`, `word_count`, `date`, `country`, `source`, `url`, `title`  --  corresponding with NOW articles
        - `{model}_summary` for each model in [PreSumm, PEGASUS, BART, ProphetNet]  -- the original summaries for $e_1$
            - `{model}_summary_len` -- word count of summary
        - `{e_2}_replaced_{model}_summary` for each model in [PreSumm, PEGASUS, BART, ProphetNet], for each $e_2 \in $[Trump, Biden, Obama, Bush]  --  the replaced summaries
            - `{e_2}_replaced_{model}_summary_len` -- word count of summary
        - `{e_2}_restored_replaced_{model}_summary` for each model in [PreSumm, PEGASUS, BART, ProphetNet], for each $e_2 \in $[Trump, Biden, Obama, Bush]  --  the replaced summaries with $e_1$ restored in the summary text, used for similarity comparisons
        - `{e_2}_{model}_summary_diff_ratio` for each model in [PreSumm, PEGASUS, BART, ProphetNet], for each $e_2 \in $[Trump, Biden, Obama, Bush] -- similarity score between the original $e_1$ summary and $e_2$ summary for a model



- `corpus_counter` (e.g., "{e_1}_public_corpus_counter.pkl")
    - a list of frequency dictionaries (Counters) of tokens for each category of summary, each element corresponds to an article
    - each row contains Counters for the following fields, descriptions are same as above:
        - `textID`
        -  `{model}_summary` for each model in [PreSumm, PEGASUS, BART, ProphetNet] 
        - `{e_2}_replaced_{model}_summary` for each model in [PreSumm, PEGASUS, BART, ProphetNet], for each $e_2 \in $[Trump, Biden, Obama, Bush] 



## Code

- load data

```
import utils

# Define source (e_1) and target (e_2) names
source_names = ["Trump", "Biden"]
target_names = ["Trump", "Biden", "Obama", "Bush"]

all_data, corpus_counters = utils.load_data(source_names)

for source in source_names:
    # corresponds to `data` described above
    data = all_data[source_name]  
    # corresponds to `corpus_counter` described above
    corpus_counter = corpus_counters[source_name]

```

- reproduce paper results
    - Run the script `code/get_results.py` (example in `code/run_results.sh`)



## Citation
```
@inproceedings{zhou-tan-2023-entity,
    title = "Entity-Based Evaluation of Political Bias in Automatic Summarization",
    author = "Zhou, Karen  and
      Tan, Chenhao",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.696",
    doi = "10.18653/v1/2023.findings-emnlp.696",
}
```

Contact: karenzhou@uchicago.edu 
