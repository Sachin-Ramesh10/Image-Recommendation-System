
import requests
from bs4 import BeautifulSoup

def getBnames(url):
    count = 0
    bnames = {}
    burl = url+"boards/"
    r = requests.get(burl)
    html_content = r.text
    soup = BeautifulSoup(html_content, "html5lib")
    p = soup.find_all('h3')
    q = soup.find_all('a',href = True, rel = True )
    for i in q:
        count = count + 1
        bnames[count] = i['href'].split('/')[-2]
    print(p[0].text)
    print(bnames)
    name = input("Select Board Names seperated with ',' \n")
    selection = name.split(',')
    a = bnames[int(selection[0])]
    b = bnames[int(selection[1])]
    c = bnames[int(selection[2])]
    d = bnames[int(selection[3])]
    #write_data(p[0].text,url,a,b,c,d)


import csv
def write_data(name,link, a, b, c,d):
    with open('boards.csv', 'a', newline='', encoding= 'utf-8') as csvfile:
        fieldnames = ['UserNames','URL', 'board1', 'board2','board3','board4']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'UserNames': name, 'URL':link, 'board1': a, 'board2': b,'board3':c,'board4':d})


text_file = open("user_queue.txt", "r")
lines = text_file.read().split('\n')

for i in lines:
   getBnames(i)
