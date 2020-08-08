from cleantext import clean
import glob
from nltk.tokenize import TweetTokenizer, word_tokenize
import numpy as np
import os
import pandas as pd
import shutil

twenty_to_six = {
    "comp.graphics": "comp",
    "comp.os.ms-windows.misc": "comp",
    "comp.sys.ibm.pc.hardware": "comp",
    "comp.sys.mac.hardware": "comp",
    "comp.windows.x": "comp",
    "rec.autos": "rec",
    "rec.motorcycles": "rec",
    "rec.sport.baseball": "rec",
    "rec.sport.hockey": "rec",
    "sci.crypt": "sci",
    "sci.electronics": "sci",
    "sci.med": "sci",
    "sci.space": "sci",
    "misc.forsale": "misc",
    "talk.politics.misc": "talk",
    "talk.politics.guns": "talk",
    "talk.politics.mideast": "talk",
    "talk.religion.misc": "religion",
    "alt.atheism": "religion",
    "soc.religion.christian": "religion",
}

def _load_vocabulary(file_name, encoding='utf-8'):
    with open(file_name, 'r') as inpf:
        vocab = frozenset({word.strip() for word in inpf})
    return vocab

def _clean_new(new_file_name, vocabulary, encoding='utf-8'):
    suffixes = ("wrote:", "writes:", "says:")
    prefixes = ("subject:", "from:", "lines:", "organization:", "distribution:","nntp-posting-host:", "reply-to:")
    lines = []
    with open(new_file_name, 'r', encoding=encoding) as ifile:
        for line in ifile:
            line = line.replace(">", "").strip().lower()
            if len(line) > 0:
                if line.startswith(prefixes) or line.endswith(suffixes):
                    continue
                else:
                    refined_line = clean(line)
                    tokenized_line = word_tokenize(refined_line, language="english")
                    cleaned_line = " ".join(filter(lambda token: token in vocabulary, tokenized_line))
                    lines.append(cleaned_line)
    return " ".join(lines)

def preprocess_20newsgroups(input_dir, file_name_vocabulary, output_dir):
    folder_sections = [
        "20news-bydate-train",
        "20news-bydate-test",
    ]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    vocabulary = _load_vocabulary(file_name_vocabulary)
    for folder_section in folder_sections:
        section = folder_section.split("-")[-1]
        df = pd.DataFrame(columns=['new', 'group'])
        for path_group in glob.glob(f"{input_dir}/{folder_section}/*"):
            group = os.path.basename(path_group)
            for path_new in glob.glob(f"{path_group}/*"):
                new = os.path.basename(path_new)
                new_str = _clean_new(path_new, vocabulary)
                if len(new_str) > 0:
                    df = df.append({'new': new_str, 'group': group}, ignore_index=True)
                else:
                    print(f"Empty new: {path_new}")
        df.to_csv(os.path.join(output_dir, f"{section}.tsv"), sep='\t', encoding='utf-8', index=False, header=True)

def _threshold_dataframe(df, threshold):
    news = df['new'].values.tolist() 
    news_len = np.array(list(map(lambda new: len(new.split()), news)))        
    mask_threshold = news_len < threshold
    filtered_df = df[mask_threshold]
    len_ = news_len[mask_threshold]
    return filtered_df

def apply_transformation_20newsgroups(data_dir, threshold=512):
    columns = ['new', 'group']
    # 20_full
    output_dir_twenty_full = os.path.join(data_dir, "20_full")
    df_train = pd.read_csv(os.path.join(output_dir_twenty_full, "train.tsv"), delimiter="\t", names=columns, skiprows=1)
    df_test = pd.read_csv(os.path.join(output_dir_twenty_full, "test.tsv"), delimiter="\t", names=columns, skiprows=1)
    # 6_full
    output_dir_six_full = os.path.join(data_dir, "6_full")
    if not os.path.exists(output_dir_six_full):
        os.makedirs(output_dir_six_full)
    df_train_simplified = df_train.copy()
    df_train_simplified["group"] = df_train["group"].apply(lambda x: twenty_to_six[x])
    df_train_simplified.to_csv(os.path.join(output_dir_six_full, f"train.tsv"), sep='\t', encoding='utf-8', index=False, header=True)
    df_test_simplified = df_test.copy()
    df_test_simplified["group"] = df_test["group"].apply(lambda x: twenty_to_six[x])
    df_test_simplified.to_csv(os.path.join(output_dir_six_full, f"test.tsv"), sep='\t', encoding='utf-8', index=False, header=True)
    # 20_simplified
    output_dir_twenty_simplified = os.path.join(data_dir, "20_simplified")
    if not os.path.exists(output_dir_twenty_simplified):
        os.makedirs(output_dir_twenty_simplified)
    df_train_th = _threshold_dataframe(df_train, threshold)
    df_train_th.to_csv(os.path.join(output_dir_twenty_simplified, f"train_{threshold}.tsv"), sep='\t', encoding='utf-8', index=False, header=True)
    df_test_th = _threshold_dataframe(df_test, threshold)
    df_test_th.to_csv(os.path.join(output_dir_twenty_simplified, f"test_{threshold}.tsv"), sep='\t', encoding='utf-8', index=False, header=True)
    # 6_simplified
    output_dir_six_simplified = os.path.join(data_dir, "6_simplified")
    if not os.path.exists(output_dir_six_simplified):
        os.makedirs(output_dir_six_simplified)
    df_train_simplified_th = _threshold_dataframe(df_train_simplified, threshold)
    df_train_simplified_th.to_csv(os.path.join(output_dir_six_simplified, f"train_{threshold}.tsv"), sep='\t', encoding='utf-8', index=False, header=True)
    df_test_simplified_th = _threshold_dataframe(df_test_simplified, threshold)
    df_test_simplified_th.to_csv(os.path.join(output_dir_six_simplified, f"test_{threshold}.tsv"), sep='\t', encoding='utf-8', index=False, header=True)