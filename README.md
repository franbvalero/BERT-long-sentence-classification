# BERT for long sentence classifiaction
BERT does not process tokenized sequences of text with more than 512 word pieces, it has to truncate them. 

In the case of corpus like [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/), this represents a problem, because, it has a lot of extensive examples.

## Recurrences Over BERT (RoBERT)
In this project, we implemente the approach proposed in this article [Hierarchical Transformers for Long Document Classification](https://arxiv.org/pdf/1910.10781.pdf). 

RoBERT can process tokenized sequences of text for every size:
1. Segements the text sequence in segments of N tokens.
2. Tokenize all the segments.
3. Process with BERT all the segments.
4. The representation of BERT obtained for each segments is located in tensor.
5. The tensor will be processed by LSTM.
6. The representation obtained in the las time step of the LSTM will be used to classify.

![](https://github.com/franborjavalero/BERT-long-sentence-classification/blob/master/images/robert.jpg?raw=true)

## Dependencies

- Python 3.7
- We need the following packages (using pip):

```
pip install pandas
pip install cleantext
pip install scikit-learn
pip install torch
pip install transformers
pip install matplotlib
pip install mlxtend
pip install seaborn
pip install Unidecode
pip install nltk
```

## Usage

The two command below use the argument `True` for downloading the 20 Newsgroups corpus (only it is necessary for the first execution of each script).

The first script uses BERT for sequence classication (BERTSC), therefore truncates the sentences.

```
./launch-experiments-20newsgroups.sh True
```

The second script uses RoBERT.
```
./launch-hierarchical-experiments-20newsgroups.sh True
```

## Results
The accuracy obtenied in the reference paper in the corpus 20 News Groups using RoBERT on the full datasets is 0.84 %. In our case, for limitations of harware, we could not feed all the segmented corpus in the GPU, for this resason, we realize an experimentation with a reduced version, where the maximum number of tokens allowed by example was 512 and 1024 tokens.

In the table below are shown the results obtained, where for the same maximum lenght, BERTSC obtained better performance than RoBERT, however, we implemented our own implementetation of RoBERT and does not follow the same approach of optimization, however, is obviuous that the LSTM degradate a bit the performance.


| MAX. LENGTH | MODEL |ACC  |
|--------|-------|-----|
| 1024   | RoBERT  |0.77 |
| 512    | BERTSC  |0.79 |
| 512    | RoBERT  |0.75 |