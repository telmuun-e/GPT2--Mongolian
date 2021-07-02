import requests
from bs4 import BeautifulSoup
import re
import os

pages = set()
links = []
m = 0
f = 0

def getLinks(pageUrl):
    global n
    global m
    global f
    global pages
    global links
    while m< 10 or links:
        try:
            html = requests.get('http://mn.wikipedia.org{}'.format(pageUrl), timeout=100)
            bs = BeautifulSoup(html.text, 'html.parser')
            text = bs.find("div", {"class":"mw-content-ltr", "id":"mw-content-text"}).text
            with open(f"../data/raw/wiki_{f}.txt", "a") as w:
                if len(text) > 30:
                     m += 1
                     w.write(str(text) + os.linesep)
                w.close()
            if m > 2000:
                f += 1
                m = 0
            for link in bs.find_all('a', href=re.compile('^(/wiki/)((?!:).)*$')):
                if 'href' in link.attrs:
                    link = link.attrs['href']
                    if link not in pages:
                        links.append(link)
                        pages.add(link)
            pageUrl = links.pop(0)
            print(f"{f}-{m}: {pageUrl}")

        except requests.exceptions.HTTPError as errh:
            print("Http Error:",errh)
        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:",errc)
        except requests.exceptions.Timeout as errt:
            print("Timeout Error:",errt)
        except requests.exceptions.RequestException as err:
            print("OOps: Something Else",err)

if __name__ == "__main__":

    getLinks('/wiki/%D0%A3%D0%98%D0%A5-%D1%8B%D0%BD_%D0%B3%D0%B8%D1%88%D2%AF%D2%AF%D0%BD')

