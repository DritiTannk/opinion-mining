import pandas as pd


def extract_imdb_data():
    """
    This method handles imdb dataset.
    """
    ds = pd.read_csv('Assets/original_datasets/imdb_ds/Test.csv')
    ds1 = pd.read_csv('Assets/original_datasets/imdb_ds/Train.csv')
    ds2 = pd.read_csv('Assets/original_datasets/imdb_ds/Valid.csv')

    text_frame = [ds['text'], ds1['text'], ds2['text']]
    frames = [ds, ds1, ds2]

    merged_text_df = pd.concat(text_frame)
    merged_imd_df = pd.concat(frames)

    merged_text_df.to_csv('Assets/input/imdb_ds.csv', index=False)
    merged_imd_df.to_csv('Assets/input/imdb_ds_test.csv', index=False)


def extract_airlines_data():
    """
    This method handles twitter airlines reviews dataset.
    """
    airline_ds = pd.read_csv('Assets/original_datasets/twitter_airlines_ds/Tweets.csv')

    only_tweets = airline_ds['text']
    only_tweets.to_csv('Assets/input/airline_ds.csv', index=False)

    airline_ds.loc[airline_ds.airline_sentiment == 'positive', 'airline_sentiment'] = 1
    airline_ds.loc[airline_ds['airline_sentiment'] == 'negative', 'airline_sentiment'] = -1
    airline_ds.loc[airline_ds['airline_sentiment'] == 'neutral', 'airline_sentiment'] = 0

    all_tweets = airline_ds[['text', 'airline_sentiment']]

    all_tweets.to_csv('Assets/input/airline_ds_test.csv', index=True, index_label='SR_NO')


if __name__ == '__main__':

    extract_imdb_data()
    extract_airlines_data()





