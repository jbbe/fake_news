import pandas as pd
import numpy as np


def load_frame():
    df = pd.read_csv('snopes.csv')
# columns=['title', 'truth_val', 'real_url', 'clean_tokenized_content'] 

    return df