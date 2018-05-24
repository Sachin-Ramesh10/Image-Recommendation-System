import urllib.request
import os
from bs4 import BeautifulSoup
from selenium import webdriver
import time
from multiprocessing import Pool
import csv

urls = {}
with open('boards.csv', newline='', encoding="ISO-8859-1") as f:
    reader = csv.reader(f)
    for row in reader:
        url = []
        url.append(row[1] + row[2])
        url.append(row[1] + row[3])
        url.append(row[1] + row[4])
        url.append(row[1] + row[5])
        urls[row[0]] = url

lofu = list(urls.keys())
ucount = 0


def createdir(dirname):
        os.makedirs(dirname)

def savetrainImages(username, boardname, num, link):

    datapath = 'F:/Data/Train'
    userpath = datapath + '/' + username
    bpath = userpath + '/' + boardname
    if not os.path.exists(userpath):
        createdir(userpath)
    if not os.path.exists(bpath):
        createdir(bpath)
    i = str(num) + '.jpg'
    path = bpath + '/' + i
    try:
        urllib.request.urlretrieve(link, path)
    except Exception:
        pass

def savetestImages(username, boardname, num, link):

    datapath = 'F:/Data/Test'
    userpath = datapath + '/' + username
    bpath = userpath + '/' + boardname
    if not os.path.exists(userpath):
        createdir(userpath)
    if not os.path.exists(bpath):
        createdir(bpath)
    i = str(num) + '.jpg'
    path = bpath + '/' + i
    try:
        urllib.request.urlretrieve(link, path)
    except Exception:
        pass

def saveCVImages(username, boardname, num, link):

    datapath = 'F:/Data/CrossValidation'
    userpath = datapath + '/' + username
    bpath = userpath + '/' + boardname
    if not os.path.exists(userpath):
        createdir(userpath)
    if not os.path.exists(bpath):
        createdir(bpath)
    i = str(num) + '.jpg'
    path = bpath + '/' + i
    try:
        urllib.request.urlretrieve(link, path)
    except Exception:
        pass

def url_scrp(links, uname):

    for link in links:
        count = 0
        #uname = lofu[ucount]
        driver = webdriver.Chrome()
        driver.set_window_position(-2000, 0)
        driver.get(link)
        for i in range(0,5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)
        page_source = driver.page_source
        soup = BeautifulSoup(page_source)
        imglinks =  soup.find_all('img')
        bname = link.split('/')[-1]
        driver.close()
        if len(imglinks) > 63:
            for i in imglinks[:50]:
                count = count + 1
                savetrainImages(uname, bname, count,  i['src'])
            for i in imglinks[50:56]:
                count = count + 1
                savetestImages(uname, bname, count,  i['src'])
            for i in imglinks[56:62]:
                count = count + 1
                saveCVImages(uname, bname, count,  i['src'])

if __name__ == '__main__':
    #p = Pool(5)
    for i in lofu:
        #p.map(url_scrp, urls[i])
        url_scrp(urls[i], i)

