import re
import string

import pandas as pd

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.corpus import sentiwordnet as swn

from sklearn.metrics import accuracy_score


def clean(text):
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text


def token_stop_pos(text):
    pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}

    tags = pos_tag(word_tokenize(text))

    tag_list = []

    for word, tag in tags:
        if word.lower() not in stop_list and word.lower() not in punchs:
            tag_list.append(tuple([word, pos_dict.get(tag[0])]))

    return tag_list


def sentiment_analysis(pos_data):
    lemmatizer = WordNetLemmatizer()
    sentiment = 0
    tokens_count = 0
    lemma = ''
    synsets = []

    for word, post in pos_data:
        if post:
            lemma = lemmatizer.lemmatize(word, pos=post)

        if lemma:
            synsets = wordnet.synsets(lemma, pos=post)

        if synsets:
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1

    if not tokens_count:
        return 0
    if sentiment > 0:
        return 1
    if sentiment <= 0:
        return 0


def check_accuracy():
    """
    This method checks the accuracy of the sentiment score.
    """
    df = pd.read_csv('Assets/output/imd_snw_analysis.csv')
    df1 = pd.read_csv('Assets/input/imdb_ds_test.csv')

    new_df = df1.merge(df, on='text', left_index=True)

    column_names = ['text', 'label', 'sentiwordnet_result']

    new_df.reindex(columns=column_names)
    new_df.to_csv('Assets/output/imdb_sent_analysis.csv', index=True, index_label='SR_NO')

    accuracy = accuracy_score(new_df['label'], new_df['sentiwordnet_result'])
    print('\n\n IMDB Dataset Accuracy ==> ', accuracy)


if __name__ == '__main__':
    stop_list = stopwords.words('english')
    stop_list.append('br')
    punchs = list(string.punctuation)

    ds = pd.read_csv('Assets/input/imdb_ds.csv')

    text_data = ds['text']

    for row in text_data:
        row_index = ds.index[ds.text == row]
        len_row_index = len(row_index)
        row_clean = clean(row)
        row_post = token_stop_pos(row_clean)
        sent = sentiment_analysis(row_post)

        if len_row_index > 0:
            for idx in row_index:
                ds.loc[ds.index[idx], 'sentiwordnet_result'] = sent
        else:
            ds.loc[ds.index[row_index], 'sentiwordnet_result'] = sent

    ds.to_csv('Assets/output/imd_snw_analysis.csv', index=False)

    check_accuracy()


