from tqdm import tqdm
import os
import utils
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import re
from pathlib import Path
import numpy as np
from collections import Counter
from operator import itemgetter

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--proj_dir',
                    type=str,
                    help="project directory")
parser.add_argument('--start_yr', 
                    type=int, 
                    default=20,
                    help="start year e.g., 20") 
parser.add_argument('--end_yr', 
                    type=int, 
                    default=21,
                    help="end year e.g., 21")
parser.add_argument('--fext',
                    type=str,
                    default="all",
                    help="output file extension")

args = parser.parse_args()


PROJ_DIR = args.proj_dir
START_YR = args.start_yr
END_YR = args.end_yr
FEXT = args.fext

DATA_DIR = f"{PROJ_DIR}/data"
OUT_PATH = f"{PROJ_DIR}/out/{FEXT}"
Path(OUT_PATH).mkdir(parents=True, exist_ok=True)

source_names = ["Trump", "Biden"]
target_names = ["Trump", "Biden", "Obama", "Bush"]

summ_fields = ["bart_summary", "pegasus_cnn_summary", "prophetnet_summary", "presumm"]
orig_fields =  summ_fields + ["text"]
replaced_fields = ["replaced_%s" % v for v in orig_fields]
pair_fields = orig_fields + replaced_fields



def bayes_compare_language_from_counter_and_freq(c1, c2, prior=.01, filter_short=True):
    '''
    Fighting Words algorithm for comparing two corpora. See Monroe et al. 2008.

    Arguments:
    - c1, c2; counters from each sample
    - prior; either a float describing a uniform prior, or a dictionary describing a prior
    over vocabulary items. If you're using a predefined vocabulary, make sure to specify that
    when you make your CountVectorizer object.

    Returns:
    - A list of length |Vocab| where each entry is a (n-gram, zscore) tuple.'''
    vocab = set(c1.keys()) | set(c2.keys())
    if type(prior) is float:
        priors = {w: prior for w in vocab}
    else:
        priors = prior
    z_scores = {}
    a0 = sum(priors.values())
    n1 = sum(c1.values())
    n2 = sum(c2.values())
    print("Comparing language...")
    for w in vocab:
        # compute delta
        w1, w2, wp = c1.get(w, 0), c2.get(w, 0), priors[w]
        term1 = np.log((w1 + wp) / (n1 + a0 - w1 - wp))
        term2 = np.log((w2 + wp) / (n2 + a0 - w2 - wp))        
        delta = term1 - term2
        # compute variance on delta
        var = 1. / (w1 + wp) + 1. / (w2 + wp)
        # store final score
        z_scores[w] = delta / np.sqrt(var)
    return_list = [(w, z_scores[w], c1.get(w, 0), c2.get(w, 0)) for w in vocab]
    return_list.sort(key=itemgetter(1))
    if filter_short:
        return_list = [(w, s, cw1, cw2) for (w, s, cw1, cw2) in return_list if len(w) > 1]
    return return_list


def admin_analysis(all_data, corpus_counters, src_names, tgt_names, show_means=False):
    '''
    Performs t-tests on the number of times "administration" appears in the summaries
    for each model and formats the results into a LaTex table.
    '''
    def count_admin(fields, data, ctr):
        res = []
        
        for i, row in enumerate(tqdm(data)):
            yr = row["date"][:-6]
            row_res = {"date": row["date"], 
                        "year": yr}
            for field in fields:
                cnts = ctr[i][field]
                if "text" in field:
                    sum_len = row["word_count"]
                else:
                    sum_len = row[f"{field}_len"]

                admin = cnts.get("administration", 0)
                

                row_res[field] = admin / sum_len
                
            res.append(row_res)
        return res


    def collapse_admin_for_ttest(admin, fields):
        res = {"20": {
                f: [] for f in pair_fields

                },
            "21": {
                f: [] for f in pair_fields
            }}
        for row in admin:
            year = row["year"]
            for res_field, ctr_field in zip(pair_fields, fields):
                res[year][res_field].append(row[ctr_field])
        return res


    def make_arrows_lex(cnt_lst, show_means):
        pre_df = {f: {
        } for f in summ_fields}
        for k in cnt_lst.keys():
            cnt = cnt_lst[k]
            for f in summ_fields:
                ttest = scipy.stats.ttest_rel(cnt["replaced_"+f], cnt[f])
                p = ttest.pvalue
                stat = ttest.statistic

                num_arrows = 0
                if p < 1e-20:
                    num_arrows = 4
                elif p < 0.001:
                    num_arrows = 3
                elif p < 0.01:
                    num_arrows = 2
                elif p < 0.05:
                    num_arrows = 1

                direx = r"$\uparrow$" if stat > 0 else r"$\downarrow$"
                if num_arrows == 0:
                    direx = r"\textemdash"
                    num_arrows = 1
                if show_means:
                    orig_mean = np.mean(cnt[f])
                    rep_mean = np.mean(cnt["replaced_"+f])
                    pre_df[f][k] = r" $\to$ ".join([f"{orig_mean: .2e}", f"{rep_mean: .2e}"])
                else:
                    pre_df[f][k] = direx * num_arrows

        return pre_df

    pre_dfs = {}
    cnt_dfs = {}
    admins = {}
    admins_col = {}  
    for src_name in src_names:
        data = all_data[src_name]
        corpus_counter = corpus_counters[src_name]
        for tgt_name in tgt_names:
            if src_name == tgt_name: continue
            f_abbrev = f"{src_name.lower()}_{tgt_name.lower()}"
            fields = orig_fields + [f"{tgt_name.lower()}_{f}" for f in replaced_fields]
            admins[f_abbrev] = count_admin(fields, data, corpus_counter)
            admins_col[f_abbrev] = collapse_admin_for_ttest(admins[f_abbrev], fields)
            pre_dfs[f_abbrev] = make_arrows_lex(admins_col[f_abbrev], show_means)
            cnt_dfs[f_abbrev] = pd.DataFrame.from_dict(pre_dfs[f_abbrev], orient="index")

    combo_df = pd.concat(list(cnt_dfs.values()),
        axis=1, keys = list(cnt_dfs.keys()))

    means_ext = "_means" if show_means else ""
    with open(os.path.join(OUT_PATH, f"admin_20-21_{FEXT}{means_ext}.txt"), "w") as f:
        with pd.option_context("max_colwidth", 1000):
            f.write(combo_df.to_latex(escape=False))

    
def vp_analysis(all_data, src_names, tgt_names, show_means=False, ex_former=False):
    '''
    Performs t-tests on the number of times "vice president" appears in the summaries
    for each model and formats the results into a LaTex table.
    '''
    def count_vice(fields, data):
        res = []
        for i, row in enumerate(tqdm(data)):
            yr = row["date"][:-6]
            row_res = {"date": row["date"], 
                        "year": yr}
            for field in fields:
                text = row[field]
                if "text" in field:
                    sum_len = row["word_count"]
                else:
                    sum_len = row[f"{field}_len"]

                if ex_former:
                    vp = len(re.findall(r"(?<!former )vice president\b ((.+?)\b)", 
                                    text.lower()))
                else:
                    vp = len(re.findall(r"vice president\b ((.+?)\b)", 
                                    text.lower()))
                
                row_res[field] = vp / sum_len
                
            res.append(row_res)
        return res

    def collapse_vice_for_ttest(vice, fields):
        res = {"20": {
                f: [] for f in pair_fields

                },
            "21": {
                f: [] for f in pair_fields
            }}
        for row in vice:
            year = row["year"]
            for res_field, ctr_field in zip(pair_fields, fields):
                if ctr_field in row:
                    res[year][res_field].append(row[ctr_field]) 
        return res

    def make_arrows_lex(cnt_lst, show_means):
        pre_df = {f: {
        } for f in summ_fields}
        for k in cnt_lst.keys():
            cnt = cnt_lst[k]
            for f in summ_fields:
                ttest = scipy.stats.ttest_rel(cnt["replaced_"+f], cnt[f])
                p = ttest.pvalue
                stat = ttest.statistic

                num_arrows = 0
                if p < 1e-20:
                    num_arrows = 4
                elif p < 0.001:
                    num_arrows = 3
                elif p < 0.01:
                    num_arrows = 2
                elif p < 0.05:
                    num_arrows = 1

                direx = r"$\uparrow$" if stat > 0 else r"$\downarrow$"
                if num_arrows == 0:
                    direx = r"\textemdash"
                    num_arrows = 1
                
                if show_means:
                    orig_mean = np.mean(cnt[f])
                    rep_mean = np.mean(cnt["replaced_"+f])
                    pre_df[f][k] = r" $\to$ ".join([f"{orig_mean: .2e}", f"{rep_mean: .2e}"])
                else:
                    pre_df[f][k] = direx * num_arrows

        return pre_df

    pre_dfs = {}
    cnt_dfs = {}
    vices = {}
    vices_col = {}  
    for src_name in src_names:
        data = all_data[src_name]
        for tgt_name in tgt_names:
            if src_name == tgt_name: continue
            f_abbrev = f"{src_name.lower()}_{tgt_name.lower()}"
            fields = orig_fields + [f"{tgt_name.lower()}_{f}" for f in replaced_fields]

            vices[f_abbrev] = count_vice(fields, data)
            vices_col[f_abbrev] = collapse_vice_for_ttest(vices[f_abbrev], fields)
            pre_dfs[f_abbrev] = make_arrows_lex(vices_col[f_abbrev], show_means)
            cnt_dfs[f_abbrev] = pd.DataFrame.from_dict(pre_dfs[f_abbrev], orient="index")

    combo_df = pd.concat(list(cnt_dfs.values()),
        axis=1, keys = list(cnt_dfs.keys()))
    
    means_ext = "_means" if show_means else ""
    with open(os.path.join(OUT_PATH, f"vice_20-21_{FEXT}{means_ext}.txt"), "w") as f:
        with pd.option_context("max_colwidth", 1000):
            f.write(combo_df.to_latex(escape=False))


def plot_similarity_separate(all_data, src_names, tgt_names):
    '''
    Plot the similarity scores for each model for each source name.
    '''
    sns.set_style("whitegrid", {'axes.grid' : False})
    font = { 'weight': 'normal',
    'size'   : 16}
    matplotlib.rc('font', **font)
    c = sns.color_palette("pastel").as_hex()
    p = {'Biden': c[0], 'Obama': c[0],  'Trump':c[3], "Bush": c[3]}
    
    ratio_fields = ["%s_diff_ratios" % v for v in summ_fields]
    field_label_map = {"presumm_diff_ratios": "PreSumm", 
                       "pegasus_cnn_summary_diff_ratios": "PEGASUS",  
                       "bart_summary_diff_ratios": "BART", 
                       "prophetnet_summary_diff_ratios": "ProphetNet"}

    h = 6
    fig, axes = plt.subplots(1, len(src_names), figsize=(len(src_names) * h, 2*h/3), sharey=True)

    sims = []
    plots = []
    for i, src_name in enumerate(src_names):
        src_sims = []
        for tgt_name in tgt_names:
            if src_name == tgt_name: continue

            data_df = pd.DataFrame(all_data[src_name])
            for field in ratio_fields:
                for val in data_df[f"{tgt_name.lower()}_{field}"]:
                    fname = field_label_map.get(
                    field, 
                    field[:field.index("_")]
                    )
                    src_sims.append((fname, val, f"{tgt_name}", "all"))
                    sims.append((fname, val, f"{src_name} → {tgt_name}", "all"))

        src_sims_df = pd.DataFrame(src_sims)
        src_sims_df.columns = ["field", "Similarity", "label", "topic"]
        b = sns.barplot(x="field", y="Similarity", hue="label", data = src_sims_df, 
                    ax=axes[i],
                    order=["PreSumm", "PEGASUS", "BART", "ProphetNet"],
                    palette=p 
        )
        plots.append(b)

        axes[i].set(title=f"$e_1$: {src_name}")
        axes[i].set(ylabel=f"Similarity" if i == 0 else None, xlabel=None)

    hatches = ["//",  "++", "\\\\",]
    labels = ["PreSumm", "PEGASUS", "BART", "ProphetNet"]
    # Loop over the bars
    for b in plots:
        for bars, hatch in zip(b.containers, hatches):
            # Set a different hatch for each group of bars
            for bar in bars:
                bar.set_hatch(hatch)
        b.set_xticklabels(labels, rotation=-30, ha='left', rotation_mode='anchor')

    for ax in axes:
        ax.set(ylim=(0, 1))
        ax.legend(title=None)
    plt.savefig(os.path.join(OUT_PATH, f"separated_full_replacement_similarities_{FEXT}.pdf"),
                 bbox_inches="tight")


def entity_freq_table(corpus_counters, src_names, tgt_names, tgt_fields, show_means=False):
    '''
    Performs t-tests on the entity frequencies for each model
    and formats the results into a LaTex table.
    '''
    def get_freqs(corpus_counter, src_name, tgt_name):
        fields = summ_fields + tgt_fields[src_name]
        freqs = {f: [] for f in fields}

        for field in summ_fields:
            repf = f"{tgt_name.lower()}_replaced_{field}"
            rows = [row[field] for row in corpus_counter]
            rep_rows = [row[repf] for row in corpus_counter]

            for i in tqdm(range(len(rows))):
                num = rows[i].get(src_name.lower(), 0)
                f_len = sum([v for v in rows[i].values()])
                freqs[field].append(num/ f_len)
                
                num = rep_rows[i].get(tgt_name.lower(), 0)
                repf_len = sum([v for v in rep_rows[i].values()])
                freqs[repf].append(num/ repf_len)
        return freqs


    def make_arrows_orig(data, tgt_name, show_means):
        pre_df = {f: None for f in summ_fields}
        for f in summ_fields:
            repf = f"{tgt_name.lower()}_replaced_{f}"
            x = data[f]
            y = data[repf]

            ttest = scipy.stats.ttest_rel(y, x)
            p = ttest.pvalue
            stat = ttest.statistic

            num_arrows = 0
            if p < 1e-20:
                num_arrows = 4
            elif p < 0.001:
                num_arrows = 3
            elif p < 0.01:
                num_arrows = 2
            elif p < 0.05:
                num_arrows = 1

            direx = r"$\uparrow$" if stat > 0 else r"$\downarrow$"
            if num_arrows == 0:
                direx = r"\textemdash"
                num_arrows = 1

            if show_means:
                orig_mean = np.mean(x)
                rep_mean = np.mean(y)
                pre_df[f] = r" $\to$ ".join([f"{orig_mean: .2e}", f"{rep_mean: .2e}"])
            else:
                pre_df[f] = direx * num_arrows

        return pre_df

    freqs = {}
    pre_df = {}
    for src_name in src_names:
        corpus_counter = corpus_counters[src_name]
        for tgt_name in tgt_names:
            if src_name == tgt_name: continue
            f_abbrev = f"{src_name.lower()}_{tgt_name.lower()}"
            freqs[f_abbrev] = get_freqs(corpus_counter, src_name, tgt_name)
            pre_df[f"{src_name} -> {tgt_name}"] = make_arrows_orig(freqs[f_abbrev], tgt_name, show_means)

    cnt_df = pd.DataFrame(pre_df)
    means_ext = "_means" if show_means else ""
    with open(os.path.join(OUT_PATH, f"freqs_{FEXT}{means_ext}.txt"), "w") as f:
        with pd.option_context("max_colwidth", 1000):
            print(cnt_df.to_latex(escape=False), file=f)   


def sum_lens_ttest(all_data, src_names, tgt_names, show_means=False):
    '''
    Performs t-tests on the summary lengths for each model
    and formats the results into a LaTex table.
    '''

    def make_arrows_orig(data, tgt_name, show_means):
        pre_df = {f: None for f in summ_fields}
        for f in summ_fields[:4]:
            repf = f"{tgt_name.lower()}_replaced_{f}"
            x = [row[f+"_len"] for row in data]
            y = [row[repf+"_len"] for row in data]

            ttest = scipy.stats.ttest_rel(y, x)
            p = ttest.pvalue
            stat = ttest.statistic

            num_arrows = 0
            if p < 1e-20:
                num_arrows = 4
            elif p < 0.001:
                num_arrows = 3
            elif p < 0.01:
                num_arrows = 2
            elif p < 0.05:
                num_arrows = 1

            direx = r"$\uparrow$" if stat > 0 else r"$\downarrow$"
            if num_arrows == 0:
                direx = r"\textemdash"
                num_arrows = 1

            if show_means:
                orig_mean = np.mean(x)
                rep_mean = np.mean(y)
                pre_df[f] = r" $\to$ ".join([f"{orig_mean: .2f}", f"{rep_mean: .2f}"])
            else:
                pre_df[f] = direx * num_arrows

        return pre_df

    pre_df = {}
    for src_name in src_names:
        data = all_data[src_name]
        for tgt_name in tgt_names:
            if src_name == tgt_name: continue
            pre_df[f"{src_name} -> {tgt_name}"] = make_arrows_orig(data, tgt_name, show_means)

    cnt_df = pd.DataFrame(pre_df)
    means_ext = "_means" if show_means else ""
    with open(os.path.join(OUT_PATH, f"sum_lens_{FEXT}{means_ext}.txt"), "w") as f:
        with pd.option_context("max_colwidth", 1000):
            print(cnt_df.to_latex(escape=False), file=f)


def plot_sum_lengths_separate(all_data, src_names, tgt_names):
    '''
    Plots the summary lengths for each model for each source name.
    '''
    font = { 'weight': 'normal',
    'size'   : 16}
    matplotlib.rc('font', **font)

    
    def compare_lens(src_name, data, direx):
        summ_pairs = list(zip(orig_fields, 
                              [f"{direx.lower()}_replaced_{v}" for v in orig_fields]))
        res = []
        for row in tqdm(data):
            for before, after in summ_pairs[:-1]:
                if before == "text": continue

                row_res = [{"label": direx, "type": "original"}, 
                    {"label": direx, "type": "replaced"} ]
                for tmp in row_res:
                    tmp["field"] = before
                row_res[0]["value"] = row[before+"_len"]
                row_res[1]["value"] = row[after+"_len"]
                res += row_res
        return res

    
    lens = {}
    for src_name in src_names:
        data = all_data[src_name]
        all_lens = []

        for tgt_name in tgt_names:
            if src_name == tgt_name: continue
            f_abbrev = f"{src_name.lower()}_{tgt_name.lower()}"
            lens[f_abbrev] = compare_lens(src_name, data, tgt_name)
            all_lens += lens[f_abbrev]


        all_df = pd.DataFrame(all_lens)

        sns.set_style("whitegrid", {'axes.grid' : False})
        b = sns.catplot(x="field", y="value", hue="type",
                        col="label", data = all_df, kind="bar",
                        order=["presumm", 
                        "pegasus_cnn_summary",
                        "bart_summary",
                        "prophetnet_summary"],
                    height=4, aspect=.8,
                    palette=sns.color_palette(['mediumpurple', 'sandybrown']),
                        legend=True
                    )

        hatches = ["//", "\\\\"]
        # Loop over the bars
        for ax in b.axes.flatten():
            for bars, hatch in zip(ax.containers, hatches):
                # Set a different hatch for each group of bars
                for bar in bars:
                    bar.set_hatch(hatch)
                
        b.set(xlabel=None)
        b.set(ylabel="summary length")
        labels = ["PreSumm", "PEGASUS", "BART", "ProphetNet"]
        sns.move_legend(b, bbox_to_anchor=(.5, 1), loc="lower center", ncol=2, title=f"$e_1$: {src_name}")
        b.set_xticklabels(labels, rotation=-30, ha='left', rotation_mode='anchor')
        b.set_titles("{col_name}")
        plt.ylim(40, 100)
        plt.savefig(os.path.join(OUT_PATH, f"{src_name.lower()}_summary_lengths_40_{FEXT}.pdf"),
                    bbox_inches="tight")


def plot_articles_hist(all_data, src_names):
    '''
    Plots the number of articles per month for each source name.
    '''
    sns.set(rc = {'figure.figsize':(10,6)})
    sns.set(font_scale = 1.35)

    m_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

    def mon_to_name(mon):
        mon_num = int(mon[-2:])
        yr = mon[:2]
        name = m_dict[mon_num]
        return f"{name} \'{yr}"

    tmps = []
    for src_name in src_names:
        data = all_data[src_name]
        tmp = pd.DataFrame(data)
        tmp["month"] = tmp["date"].apply(lambda x : x[:-3])
        tmp = tmp.groupby("month").size().to_frame()
        tmp.reset_index(inplace=True)
        tmp["mentions"] = f"{src_name} only" 
        tmps.append(tmp)

    tmps = pd.concat(tmps)
    tmps.columns = ['month', "thousand articles", 'type']
    tmps["thousand articles"] = tmps["thousand articles"]/1000
    tmps = tmps.reset_index()

    months = list(tmps["month"])
    m_labels = [mon_to_name(m) for m in months]

    sns.set_style("white")
    ax = sns.lineplot(x="month", y="thousand articles", hue="type", data=tmps, linewidth = 3,
                    palette=sns.color_palette(['lightcoral', 'skyblue']))
    ax.axvline(x="21-01", ymin= 0, ymax=tmps["thousand articles"].max(), 
            color="gray", linestyle=(0, (5, 5)),
            label="Transition of power"
            )

    n_labels = 2
    ax.legend(title=None)
    ax.set(xlabel=None)
    ax.set_xticklabels(labels=m_labels[::n_labels])
    ax.set_xticks(range(0, 24, n_labels))
    plt.xticks(rotation=-45)
    plt.savefig(f'{OUT_PATH}/hist_{FEXT}.pdf', bbox_inches="tight")


def plot_similarity_distributions(all_data, src_names, tgt_names):
    '''
    Plots the distribution of similarity scores for each model
    '''
    sns.set_style("whitegrid", {'axes.grid' : False})
    font = { 'weight': 'normal',
    'size'   : 16}
    matplotlib.rc('font', **font)
    c = sns.color_palette("pastel").as_hex()
    p = {'Biden': c[0], 'Obama': c[0],  'Trump':c[3], "Bush": c[3]}
    
    ratio_fields = ["%s_diff_ratios" % v for v in summ_fields]
    field_label_map = {"presumm_diff_ratios": "PreSumm", 
                       "pegasus_cnn_summary_diff_ratios": "PEGASUS",  
                       "bart_summary_diff_ratios": "BART", 
                       "prophetnet_summary_diff_ratios": "ProphetNet"}
    models = ["PreSumm", "PEGASUS", "BART", "ProphetNet"]

    
    for model in models:
    
        h = 6
        fig, axes = plt.subplots(1, len(src_names), figsize=(len(src_names) * h, 2*h/3), sharey=True)
        
        for i, src_name in enumerate(src_names):
            src_sims = []
            for tgt_name in tgt_names:
                if src_name == tgt_name: continue

                data_df = pd.DataFrame(all_data[src_name])
                for field in ratio_fields:
                    for val in data_df[f"{tgt_name.lower()}_{field}"]:
                        fname = field_label_map.get(
                        field, 
                        field[:field.index("_")]
                        )
                        src_sims.append((fname, val, f"{tgt_name}", "all"))

            src_sims_df = pd.DataFrame(src_sims)
            src_sims_df.columns = ["field", "Similarity", "label", "topic"]
            
            g = sns.histplot(src_sims_df[src_sims_df["field"] == model], 
                x="Similarity", hue="label", multiple="dodge",
                stat="probability", common_norm=False,
                bins= 10, 
                palette=p,
                ax=axes[i]
            )
        
            axes[i].set(title=f"$e_1$: {src_name}")
            axes[i].set(ylabel=f"Probability" if i == 0 else None, xlabel=None)

            # Define some hatches
            hatches = ["//",  "++", "\\\\",]
            legend = axes[i].get_legend()
            legend.set_title(f"$e_2$: ")
            # Loop over the bars
            for bars, hatch, handle in zip(axes[i].containers, hatches, legend.legendHandles[::-1]):
                # update the hatching in the legend handle
                handle.set_hatch(hatch)
                # Set a different hatch for each group of bars
                for bar in bars:
                    bar.set_hatch(hatch)
            plt.legend(loc='upper left', title=f"$e_2$: ", labels=[t for t in tgt_names if t != src_name])
        plt.savefig(f'{OUT_PATH}/sim_dist_{model}.pdf',
                    bbox_inches="tight")


def FW_by_similarity(all_data,  corpus_counters, src_names, tgt_names):
    '''
    Applies the Fightin Words algorithm to articles leading to high similarity summaries (HighSim) 
    vs. low similarity summaries (LowSim)

    Note: this requires access to the full article text, which must be obtained from NOW
    https://www.english-corpora.org/now/
    '''
    UPPER = 0.75
    LOWER = 0.25

    def get_new_corp_ctrs(src_name, tgt_name, data, corpus_counter, upper_qs, lower_qs):
        up_ctrs = {f: Counter() for f in summ_fields}  # ctrs for articles w most sim
        lw_ctrs = {f: Counter() for f in summ_fields}  # ctrs for arts w least sim

        for i, row in enumerate(data):
            for field in summ_fields:
                txt_ctrs = corpus_counter[i][f"text"]
                rf = f"{tgt_name.lower()}_{field}_diff_ratios"
                if row[rf] >= upper_qs[field]:
                    up_ctrs[field].update(txt_ctrs)
                elif row[rf] <= lower_qs[field]:
                    lw_ctrs[field].update(txt_ctrs)
        return up_ctrs, lw_ctrs
    
    upper_qs = {}
    lower_qs = {}
    up = {}
    lw = {}

    fw_names = [(n1, n2) for n1 in src_names for n2 in src_names if n1 != n2]

    for src_name, tgt_name in fw_names:
        print(src_name, tgt_name) 
        data = all_data[src_name]
        corpus_counter = corpus_counters[src_name]
        f_abbrev = f"{src_name.lower()}_{tgt_name.lower()}"
        data_df = pd.DataFrame(data)
        upper_qs[f_abbrev] = {f: None for f in summ_fields}
        lower_qs[f_abbrev] = {f: None for f in summ_fields}
    
        for field in summ_fields:
            rf = f"{field}_diff_ratios"
            print(field, rf)
            tgt_field = f"{tgt_name.lower()}_{rf}"
            if field != "presumm":
                upper_qs[f_abbrev][field] = data_df[tgt_field].quantile(UPPER)
                lower_qs[f_abbrev][field] = data_df[tgt_field].quantile(LOWER)
            else: # is presumm
                upper_qs[f_abbrev][field] = 1.0
                lower_qs[f_abbrev][field] = data_df[data_df[tgt_field] != 1][tgt_field].max()

        up[f_abbrev], lw[f_abbrev] = get_new_corp_ctrs(src_name, tgt_name, data, corpus_counter, upper_qs[f_abbrev], lower_qs[f_abbrev])


    def format_fw_freq(fwl, file, show_freqs=False, n=10):
        row1 = ""
        for w, s, c1, c2 in fwl[-n:][::-1]:
            if not show_freqs:
                row1 += "{word} ({score:.3f}), ".format(word=w, score=s, freq=c1)
            else:
                row1 += "{word} ({score:.3f}, {freq}), ".format(word=w, score=s, freq=c1)
  
        row1 = row1[:-2] +"}, "
        file.write("\ctext[RGB]{{140,189,144}}{{{row}\n".format(row=row1))

        row2 = ""
        for w, s, c1, c2 in fwl[:n]:
            if not show_freqs:
                row2 += "{word} ({score:.3f}), ".format(word=w, score=s, freq=c2)
            else:
                row2 += "{word} ({score:.3f}, {freq}), ".format(word=w, score=s, freq=c2)
        file.write("\ctext[RGB]{{255,159,121}}{{{row}}}\n".format(row=row2[:-2]))

    with open(os.path.join(OUT_PATH, f"{OUT_PATH}/sims_art_fw_no_freq_{FEXT}.txt"), "w") as f:
        for field in summ_fields:
            f.write(f"[{field}]\n")
            for src_name, tgt_name in fw_names: 
                f.write(f"{src_name} -> {tgt_name}\n")
                f_abbrev = f"{src_name.lower()}_{tgt_name.lower()}"
                fw = bayes_compare_language_from_counter_and_freq(up[f_abbrev][field], lw[f_abbrev][field])
                format_fw_freq(fw, f)
            f.write("\n")
            f.write("\n")



if __name__ == "__main__":
    
    all_data, corpus_counters = utils.load_data(source_names, 
                                                use_data=True, 
                                                use_counters=True, 
                                                data_path=DATA_DIR)

    tgt_fields = {src: [] for src in source_names}
    src_fields = {}
    tgt_restored_fields = {src: [] for src in source_names}

    for src_name in source_names:
        for tgt_name in target_names:
            if src_name == tgt_name: continue
            tgt_fields[src_name] += [f"{tgt_name.lower()}_{f}" for f in replaced_fields]


    plot_articles_hist(all_data, source_names) 

    plot_similarity_separate(all_data, source_names, target_names)

    plot_sum_lengths_separate(all_data, source_names, target_names)


    for show_means in [True, False]:

        sum_lens_ttest(all_data, source_names, target_names, show_means=show_means)

        entity_freq_table(corpus_counters, source_names, target_names, tgt_fields, show_means=show_means)

        admin_analysis(all_data, corpus_counters, source_names, target_names, show_means=show_means)

        vp_analysis(all_data, source_names, target_names, show_means=show_means)

    plot_similarity_distributions(all_data, source_names, target_names)

    FW_by_similarity(all_data, corpus_counters, source_names, target_names)