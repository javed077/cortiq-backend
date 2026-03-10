import requests

HN_TOP = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM = "https://hacker-news.firebaseio.com/v0/item/{}.json"

def get_hn_trends(limit=10):

    ids = requests.get(HN_TOP).json()[:limit]

    stories = []

    for i in ids:

        data = requests.get(HN_ITEM.format(i)).json()

        if not data:
            continue

        stories.append({
            "title": data.get("title"),
            "score": data.get("score"),
            "url": data.get("url")
        })

    return stories