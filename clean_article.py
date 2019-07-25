import nltk, re
import requests
from bs4 import BeautifulSoup
from urllib import request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



def is_fact_check():
    url = 'https://www.snopes.com/random'
    plain = request.urlopen(url).read().decode('utf8')
    line_begin = plain.find('<link rel="canonical"')
    if 'fact-check' in plain[real_url:real_url+150]:
        return plain
    return None

def process_url():
    plain = is_fact_check()
    while plain is None:
        plain = is_fact_check()
    claim_idx = plain.find('<div class="claim">') + 19
    end_claim = plain[claim_idx:].find('</div>') + claim_idx
    title = plain[claim_idx:end_claim]
    title = re.sub(r'<[/p]+>', "", title).strip()
    title = re.sub(r'\s{2,}', " ", title).strip()
    truth_val_start = plain.find('rating-label-')+13
    truth_val_end = truth_val_start + plain[truth_val_start:].find('"')
    truth_val = plain[truth_val_start:truth_val_end]

    content_start = plain.find('<div class="content">')
    article_end = content_start + plain[content_start:].find('</article>')
    content = plain[content_start:article_end]
    content = re.sub(r'<script [*] />', "", content)
    content = re.sub(r'<.*?>', "", content)
    content = re.sub(r'\s[\s]+', " ", content)
    content = re.sub(r'\(\{[.*]\}\);', "", content)
    content = re.sub(r'[{\[][.*][}\]]', "", content)
    content = re.sub(r'{}', "", content)
    content = re.sub(r'!function\(.*?\){.*?}', "", content)
    content = re.sub(r'{.*?}', "", content)
    content = re.sub(r'[})]+', "", content)
    content = re.sub(r'\(window, document, "script", "Rumble"; Rumble\("play", ', "", content)
    content = re.sub(r'freestar.queue.push\(function \( ; ', "", content)
    stops = stopwords.words('english')
    clean_tokenized_content = [w for w in list(word_tokenize(content)) if w.isalpha() and w not in stops]
    # print(clean_tokenized_content)
    return title, truth_val, clean_tokenized_content
    # content_end = plain.find('<div class="content">')

    

if __name__ == "__main__":
   outfile = 'snopes.csv'
   with open(outfile, 'w+') as f:
        for i in range(1000):
            title, truth_val, clean_tokenized_content = process_url()
            print(title)
            out_line = ', '.join([title, truth_val, clean_tokenized_content]) + '\n'
            f.write(out_line)
    





# print(process_url(url))