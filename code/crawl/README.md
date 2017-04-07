# Crawler for Elsevier articles

Howto:
* Get your API key from https://dev.elsevier.com/user/login (if not registered, you have to do that first)
* Save the API key into file API_KEY
* Run the crawler using one of two options:
    * `run_crawl.sh` - with already specified parameters to download articles used for document classification pre-step in our paper
    * `crawl.py SUBJECT_CODE YEAR` - to download articles from a specific subject and year
        * SUBJECT_CODE can be chosen from http://api.elsevier.com/content/subject/scidir?httpAccept=text/xml
