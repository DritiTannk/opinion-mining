import re
import string

import pandas as pd
import emojis as em

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.corpus import sentiwordnet as swn

from emosent import get_emoji_sentiment_rank

from sklearn.metrics import accuracy_score


def clean(text):
    text = re.sub('[.]*', '', text)
    text = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|"\
                  "(?:%[0-9a-fA-F][0-9a-fA-F]))+", '', text)

    return text


def check_emotions(text):
    """
    This method checks for the emojis in the text and return its sentiment score.
    """
    overall_sentiment = 0

    emojis_set = em.get(text)

    if len(emojis_set) != 0:
        for e in emojis_set:

            if len(e) >= 2:
                try:
                    emoji_sent = get_emoji_sentiment_rank(e[0])
                    overall_sentiment += emoji_sent.get('sentiment_score')
                except KeyError:
                    overall_sentiment += 0

    return overall_sentiment


def token_stop_pos(text):
    pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}

    tags = pos_tag(word_tokenize(text))

    tag_list = []

    for word, tag in tags:
        if word.lower() not in stop_list and word.lower() not in punchs:
            tag_list.append(tuple([word, pos_dict.get(tag[0])]))

    return tag_list


def sentiwordnetanalysis(pos_data):
    """
    This method returns the overall sentiment score of the text.
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    sentiment = 0
    tokens_count = 0
    for word, pos in pos_data:
        if not pos:
            continue
        lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
        if not lemma:
            continue
        synsets = wordnet.synsets(lemma, pos=pos)
        if not synsets:
            continue
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        tokens_count += 1

    return sentiment


def generate_csv():
    df = pd.read_csv('Assets/output/emotions_analysis.csv')
    df1 = pd.read_csv('Assets/input/airline_ds_test.csv')

    new_df = df1.merge(df, on='text', left_index=True)

    column_names = ['text', 'airline_sentiment', 'sentiwordnet_result']

    new_df.reindex(columns=column_names)
    new_df.to_csv('Assets/output/air_merged.csv')

    accuracy = accuracy_score(new_df['airline_sentiment'], new_df['sentiwordnet_result'])
    print('\n\n Airlines Dataset Accuracy ==> ', accuracy)


if __name__ == '__main__':
    stop_list = stopwords.words('english')
    stop_list.append('br')
    punchs = list(string.punctuation)
    punchs += ('”', '’', '``')

    ds = pd.read_csv('Assets/input/airline_ds.csv')

    text_data = ds['text']

    for row in text_data:
        row_index = ds.index[ds.text == row]
        len_row_index = len(row_index)

        clean_row = clean(row)
        emojis_sentiment = check_emotions(clean_row)
        row_post = token_stop_pos(clean_row)
        sent = sentiwordnetanalysis(row_post)

        overall_tweet_sent = sent + emojis_sentiment

        tweet_sent = 0

        if overall_tweet_sent > 0:
            tweet_sent = 1
        if overall_tweet_sent < 0:
            tweet_sent = -1
        if overall_tweet_sent == 0:
            tweet_sent = 0

        if len_row_index > 0:
            for idx in row_index:
                ds.loc[ds.index[idx], 'sentiwordnet_result'] = tweet_sent
        else:
            ds.loc[ds.index[row_index], 'sentiwordnet_result'] = tweet_sent

    ds.to_csv('Assets/output/emotions_analysis_5.csv', index=False)

    generate_csv()





