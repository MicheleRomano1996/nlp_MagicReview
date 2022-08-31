# librerie utili
import time
from selenium import webdriver
import pandas as pd
from transformers import pipeline
import requests
import numpy as np
from selenium.webdriver.common.keys import Keys
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# nome del luogo di interesse
name_of_place ='name of the place'


start_time_total = time.time()
start_time = time.time()


# definizione del classifier per sentiment analysis
classifier = pipeline("sentiment-analysis",model='distilbert-base-uncased-finetuned-sst-2-english')


# inizio script:

# 1) cerca il ristorante su maps e trova il numero di recensioni totali
driver = webdriver.Chrome()
url = 'https://www.google.it/maps/@45.5511399,10.2147144,14z'
driver.get(url)

# accettare coockies (obbligatorio per la prima volta)
time.sleep(2.5)
driver.find_element_by_xpath('//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[3]/div[1]/div[1]/form[2]/div/div/button').click()
m = driver.find_element_by_name("q")
m.send_keys(name_of_place)
time.sleep(0.5)
m.send_keys(Keys.ENTER)
time.sleep(5.5)


# calcolo del numero totale di recensioni
total_number_of_reviews = driver.find_element_by_xpath('//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[1]/div[1]/div[2]/div/div[1]/span[1]/span/span/span[2]/span[1]/button').text.split(" ")[0]
if ',' in total_number_of_reviews:
    total_number_of_reviews = int(total_number_of_reviews.replace(',',''))
elif '.' in total_number_of_reviews:
    total_number_of_reviews = int(total_number_of_reviews.replace('.',''))
else:
    total_number_of_reviews = int(total_number_of_reviews)
print(f'total number of review: {total_number_of_reviews}')
print()


finish_time = time.time()
delta_time = finish_time - start_time
print(f'time: {round(delta_time,2)} s')

print('__________________________________\n')


# 2) caricamento recensioni totali, 
start_time = time.time()

# cliccare su recensioni
time.sleep(2)
driver.find_element_by_xpath('//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[1]/div[1]/div[2]/div/div[1]/span[1]/span/span/span[2]/span[1]/button').click()

total_number = int(total_number_of_reviews/10+1)



# loop per caricare altre recensioni
last_height = driver.execute_script("return document.body.scrollHeight")   # Get scroll height
number = 0  # counter for the loop

while True:   
    number = number+1

    # Scroll down to bottom -> Find scroll layout
    ele = driver.find_element_by_xpath('//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]')
    driver.execute_script('arguments[0].scrollBy(0, 5000);', ele)

    # Wait to load page
    time.sleep(1.5)

    # Calculate new scroll height and compare with last scroll height
    ele = driver.find_element_by_xpath('//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]')
    new_height = driver.execute_script("return arguments[0].scrollHeight", ele)


    if number == total_number:
        break
    if new_height == last_height:
        break
        
    last_height = new_height

    
# find review xpath for expand "Altro/More" per espandere tutte le recensioni
item = driver.find_elements_by_xpath('//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[10]')
time.sleep(1)

finish_time = time.time()
delta_time = finish_time - start_time
print(f'time: {round(delta_time,2)} s\n')

print(f'Scraping finished!\n')
print('__________________________________\n')



# 2) creazione dataframe contenente data, stelle e testo della recensione
start_time = time.time()
date_list = []
star_list = []
review_text_list = []

# loop to extract review and other 
for i in item:
    button = i.find_elements_by_tag_name('button')
    for m in button:
        if m.text == "Altro":
            m.click()
        time.sleep(0.2)
    time.sleep(0.7)

    stars = i.find_elements_by_class_name("kvMYJc")
#     review = i.find_elements_by_class_name("wiI7pd")
    review = i.find_elements_by_class_name("MyEned")
    date = i.find_elements_by_class_name("rsqaWe")

    for k,l,p in zip(stars,review,date):
        star_list.append(k.get_attribute("aria-label"))
        review_text_list.append(l.text)
        date_list.append(p.text)

driver.quit()

new_text = []

# nlp pre processing cleaning
for i in range(len(review_text_list)):
    prova_text = review_text_list[i]
    prova_text = prova_text.replace('\n',' ')
    prova_text = prova_text.replace('...','. ')
    prova_text = prova_text.replace('..','. ')
    prova_text = prova_text.replace('"','')
    prova_text = prova_text.replace('(Traduzione di Google)','')
    prova_text = prova_text.replace('(Originale)','')
    new_text.append(prova_text)

new_df = pd.DataFrame(
    {'Date': date_list,
     'Star': star_list,
     'Text': new_text})


new_df.head()

df_analysis = new_df.copy()
for i in new_df.index:
    text_class_i = list(new_df[new_df.index == i].Text.values)
    if '' in text_class_i:
        df_analysis.drop(i,inplace=True)
        


print(f'New df finished!\nLunghezza df: {len(df_analysis)}\n')
finish_time = time.time()
delta_time = finish_time - start_time
print(f'time: {round(delta_time,2)} s')
print('__________________________________\n')


# 4) sentiment analysis: traduzione automatica in inglese e classificazione in positive/negative, con transformers di hugging face

start_time = time.time()

results = {}
text_trad_list = []

for k in range(len(df_analysis)):
    text_trad = df_analysis.Text.values[k] 
    r = requests.post(url='https://hf.space/embed/Sa-m/Auto-Translation/+/api/predict/', json={"data": [text_trad]})
    new_frase = r.json()['data'][0]
    text_trad_list.append(new_frase)
    res = classifier(new_frase)[0]['label']
    if res == 'POSITIVE':
        res = 1
    else:
        res = 0
    results[text_trad] = res

a = np.array(list(results.values()))
print('Review finished!\n')
print(f'Recensioni scritte positive: {np.count_nonzero(a > 0, axis=0)}\nRecensioni scritte negative: {np.count_nonzero(a == 0, axis=0)}\n')

# media stelle
star_list_num = []
star_list = new_df.Star.values
for j in range(len(star_list)):
    star_list[j] = star_list[j].replace('stella ','')
    star_list[j] = star_list[j].replace('stelle ','')
    star_list[j] = star_list[j].replace(' ','')
    star_list_num.append(int(star_list[j]))
mean_star = round(np.mean(star_list_num),2)
print(f'Media stelle: {mean_star} (recensioni totali: {len(new_df)})\n')

finish_time = time.time()
delta_time = finish_time - start_time
print(f'time: {round(delta_time,2)} s')
print('__________________________________\n')


finish_time_total = time.time()
delta_time_total = finish_time_total - start_time_total
print(f'tempo totale: {round(delta_time_total/60,2)}')
print('__________________________________\n')



# wordcloud plot
unique_string = (" ").join(text_trad_list)
wordcloud = WordCloud(max_font_size=50, max_words=40, background_color="white").generate(unique_string)
plt.figure(dpi=120)
plt.imshow(wordcloud, interpolation='Bicubic')
plt.axis("off")
plt.show()


# bar chart for 
# function that count the frequency of all words in all reviews
def word_count(str):
    
    # manual attempt to eliminate common words
    list_word_exclude = ['the','and','a','that','is','in','at','have','has','i','the','for','but','was','we','with','of','it'
                         ,'had','they','are','this','to','who','on','also','my','you','not','only']
    counts = dict()
    words = str.split()
    for word in words:
        word = word.lower()
        if word in list_word_exclude:
            pass
        else:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
    return counts

count_result = word_count(unique_string)
df_count = pd.Series(count_result).to_frame().reset_index()
df_count.rename(columns={"index": "Text", 0: "Count"},inplace=True)
df_count.sort_values(['Count'],ascending=False,inplace=True)
df_count_head = df_count.head(20)
df_count_head.plot.bar(x = "Text", y = 'Count')