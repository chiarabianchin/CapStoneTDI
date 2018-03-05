import os
import time
import sys
import urllib
from progressbar import ProgressBar

def get_raw_html(url):
    version = (3,0)
    curr_version = sys.version_info
    if curr_version >= version:     #If the Current Version of Python is 3.0 or above
        import urllib.request    #urllib library for Extracting web pages
        try:
            headers = {}
            headers['User-Agent'] = "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"
            request = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(request)
            respData = str(resp.read())
            return respData
        except Exception as e:
            print(str(e))
    else:                        #If the Current Version of Python is 2.x
        import urllib2
        try:
            headers = {}
            headers['User-Agent'] = "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"
            request = urllib2.Request(url, headers = headers)
            try:
                response = urllib2.urlopen(request)
            except URLError: # Handling SSL certificate failed
                context = ssl._create_unverified_context()
                response = urlopen(req,context=context)
            #response = urllib2.urlopen(req)
            raw_html = response.read()
            return raw_html
        except:
            return"Page Not found"


def next_link(s):
    start_line = s.find('rg_di')
    if start_line == -1:    #If no links are found then give an error!
        end_quote = 0
        link = "no_links"
        return link, end_quote
    else:
        start_line = s.find('"class="rg_meta"')
        start_content = s.find('"ou"',start_line+1)
        end_content = s.find(',"ow"',start_content+1)
        content_raw = str(s[start_content+6:end_content-1])
        return content_raw, end_content


def all_links(page):
    links = []
    while True:
        link, end_content = next_link(page)
        if link == "no_links":
            break
        else:
            links.append(link)      #Append all the links in the list named 'Links'
            #time.sleep(0.1)        #Timer could be used to slow down the request for image downloads
            page = page[end_content:]
    return links

def download_images(links, search_keyword, def_choice=False):

    if def_choice:
        choice = 'y'
        num = 100
    else:
        choice = raw_input("Do you want to save the links? [y]/[n]: ")
        num = raw_input("Enter number of images to download (max 100): ")

    if choice=='y' or choice=='Y':
        #write all the links into a test file.
        f = open('links.txt', 'a')        #Open the text file called links.txt
        for link in links:
            f.write(str(link))
            f.write("\n")
        f.close()   #Close the file

    counter = 1
    errors=0
    search_keyword = search_keyword.replace("%20","_")
    directory = search_keyword+'/'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    pbar = ProgressBar()
    for link in pbar(links):
        if counter<=int(num):
            file_extension = link.split(".")[-1]
            filename = directory + str(counter) + "."+ file_extension
            #print ("Downloading image: " + str(counter)+'/'+str(num))
            try:
                urllib.urlretrieve(link, filename)
            except IOError:
                errors+=1
                #print ("\nIOError on Image" + str(counter))
            except urllib.error.HTTPError as e:
                errors+=1
                #print ("\nHTTPError on Image"+ str(counter))
            except urllib.error.URLError as e:
                errors+=1
                #print ("\nURLError on Image" + str(counter))

        counter+=1
    return errors


def search():

    version = (3,0)
    curr_version = sys.version_info
    if curr_version >= version:     #If the Current Version of Python is 3.0 or above
        import urllib.request    #urllib library for Extracting web pages
    else:
        import urllib2 #If current version of python is 2.x

    search_keyword = raw_input("Enter the search query: ")

    #Download Image Links
    links = []
    search_keyword = search_keyword.replace(" ","%20")
    url = 'https://www.google.com/search?q=' + search_keyword+ '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
    raw_html =  (get_raw_html(url))
    links = links + (all_links(raw_html))
    print ("Total Image Links = "+str(len(links)))
    print ("\n")
    errors = download_images(links, search_keyword)
    print ("Download Complete.\n"+ str(errors) +" errors while downloading.")

def auto_search(list_of_searches):
    version = (3,0)
    curr_version = sys.version_info
    if curr_version >= version:     #If the Current Version of Python is 3.0 or above
        import urllib.request    #urllib library for Extracting web pages
    else:
        import urllib2 #If current version of python is 2.x

    for search_keyword in list_of_searches:
        print("Search ", search_keyword)

        #Download Image Links
        links = []
        search_keyword = search_keyword.replace(" ","%20")
        url = 'https://www.google.com/search?q=' + search_keyword+ '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
        raw_html =  (get_raw_html(url))
        links = links + (all_links(raw_html))
        print ("Total Image Links = "+str(len(links)))
        print ("\n")
        errors = download_images(links, search_keyword, def_choice=True)
        print ("Download Complete.\n"+ str(errors) +" errors while downloading.")

#search()
#auto_search(['zucchini', 'courgettes', 'zucchini images'])
#auto_search(['zucchini trombetta', 'green zucchini', 'yellow zucchini'])
#auto_search(['real eggs', 'eggs in fridge', 'egg tray'])
#auto_search(['potato', 'raw potato', 'potato bag'])
auto_search(['yellow potatoes'])
#auto_search(['tomato', 'red tomato', 'tomato in fridge'])
#auto_search(['lettuce', 'insalata', 'butterhead','crisphead', 'radicchio','white lettuce', 'rucola'])