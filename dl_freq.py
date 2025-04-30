import requests
import bs4

url = 'https://en.m.wiktionary.org/wiki/Wiktionary:Frequency_lists/Spanish1000'

url = 'https://en.m.wiktionary.org/wiki/User:Matthias_Buchmeier/Spanish_frequency_list-1-5000'

r = requests.get(url)
soup = bs4.BeautifulSoup(r.content, features='html.parser')
f = open('5000_freq.txt', 'a+')

list= []
for s in soup.find_all('p'):
    for a in s.find_all('a'):
        if not a.text in list:
            list.append(a.text)
            print(a.text)
            f.write(a.text)
print(len(list))