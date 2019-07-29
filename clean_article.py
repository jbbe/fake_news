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
    link_end = plain[line_begin:].find('" />') + line_begin
    # print(plain[line_begin:link_end])
    if 'fact-check' in plain[line_begin:link_end]:
        return plain
    return None

def process_url():
    plain = is_fact_check()
    while plain is None:
        plain = is_fact_check()
    line_begin = plain.find('<link rel="canonical"')
    link_end = plain[line_begin:].find('" />') + line_begin
    real_url = plain[line_begin+28:link_end]
    # print(real_url)
    claim_idx = plain.find('<div class="claim">') + 19
    end_claim = plain[claim_idx:].find('</div>') + claim_idx
    title = plain[claim_idx:end_claim]
    # title = re.sub(r'<[/p]+>', "", title).strip()
    title = re.sub(r'<[^>]+>', '', title).strip()
    title = re.sub(r'\s{2,}', " ", title).replace(',', '')
    truth_val_start = plain.find('rating-label-')+13
    truth_val_end = truth_val_start + plain[truth_val_start:].find('"')
    truth_val = plain[truth_val_start:truth_val_end].replace(',', '')

    content_start = plain.find('<div class="content">')
    article_end = content_start + plain[content_start:].find('</article>')
    content = plain[content_start:article_end]
    content = re.sub(r'<[^>]+>', '', content)
    # print(content)
    content = re.sub(r'<script [*]+ />', "", content)
    # content = re.sub(r'<.*?>', "", content)
    content = re.sub(r'\s[\s]+', " ", content)
    content = re.sub(r'\(\{[.*]\}\);', "", content)
    content = re.sub(r'[{\[][.*][}\]]', "", content)
    content = re.sub(r'{}', "", content)
    content = re.sub(r'!function\(.*?\){.*?}', "", content)
    content = re.sub(r'{.*?}', "", content)
    content = re.sub(r'[})]+', "", content)
    content = re.sub(r'\(window, document, "script", "Rumble"; Rumble\("play", ', "", content)
    content = re.sub(r'freestar.queue.push\(function \( ; ', "", content)
    content = re.sub(r'<link rel=".*?"', "", content)
    content = re.sub(r'href=".*?"', "", content)
    stops = stopwords.words('english')
    clean_tokenized_content = ' '.join([w for w in list(word_tokenize(content)) if w.isalpha() and w not in stops])
    # print(clean_tokenized_content)
    return title, truth_val, real_url, clean_tokenized_content
    # content_end = plain.find('<div class="content">')

    

if __name__ == "__main__":
   outfile = 'snopes.csv'
   with open(outfile, 'w+') as f:
        f.write('title, truth, url,  content\n') 
        for i in range(50):
            title, truth_val, real_url, clean_tokenized_content = process_url()
            # print(title)
            # if 'false' in truth_val:
            out_line = ', '.join(str(v) for v in [title, truth_val, real_url, clean_tokenized_content, '\n'])
            f.write(out_line)
    





# print(process_url(url))