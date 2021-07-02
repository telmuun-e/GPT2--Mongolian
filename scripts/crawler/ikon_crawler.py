import requests
from bs4 import BeautifulSoup
import re
import os

pages = set()
links = []
m = 0
f = 0

def getLinks(url):
    global m
    global f
    global pages
    global links
    while m < 10 or links:
        try:
            html = requests.get('https://ikon.mn{}'.format(url), timeout=10) 
            bs = BeautifulSoup(html.text, 'html.parser')
            try:
                text = bs.find("div", {"class":"icontent"}).text
                with open(f"../data/raw_data/ikon_{f}.txt", "a") as w:
                    if len(text) > 30:
                        m += 1
                        w.write(str(text) + os.linesep)
                    w.close()
                if m > 10000:
                    f += 1
                    m = 0
            except:
                print("no text")
            for link in bs.find_all('a', href=re.compile('^/(l|n)/[\w\d]*')):
                if 'href' in link.attrs:
                    link = link.attrs['href']
                    if link not in pages and "#" not in link:
                        links.append(link)
                        pages.add(link)
            url = links.pop(0)
            print(f"{f}-{m}: {url}")
            
        except requests.exceptions.HTTPError as errh:
            print("Http Error:",errh)
        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:",errc)
        except requests.exceptions.Timeout as errt:
            print("Timeout Error:",errt)
        except requests.exceptions.RequestException as err:
            print("OOps: Something Else",err)
            
if __name__ == "__main__":

    getLinks('/n/23o9')
