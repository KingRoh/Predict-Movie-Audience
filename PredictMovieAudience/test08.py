#import
import urllib.request
import json
import re
MovieName = input()
client_id = "2F20Z6YIPdWkrFVixfxB"
client_secret = "RbA6BI0lmt"

url = "https://openapi.naver.com/v1/search/movie.json"
option = "&display=20&sort=count"
query = "?query="+urllib.parse.quote(MovieName)
url_query = url + query + option


request = urllib.request.Request(url_query)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)


response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode == 200):
    response_body = response.read()
    result = json.loads(response_body.decode('utf-8'))
    items = result.get('items')
    #print(response_body.decode('utf-8'))
    title = re.sub('<[^<]+?>', '', items[0]['title'])

    print(result)
    #print(title)


else:
    print("Error code:"+rescode)