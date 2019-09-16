from urllib.request import urlopen
from urllib.request import HTTPError
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin


url = 'http://www.maxmovie.com/'
page = urlopen(url)
soup = BeautifulSoup(page, "html.parser")
MovieNews2 = soup.findAll('div',{'class':"tyb"})
MovieNews2
print(MovieNews2)
