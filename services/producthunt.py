import requests

PRODUCT_HUNT_URL = "https://api.producthunt.com/v2/api/graphql"

def get_producthunt_trends(token):

    query = """
    {
      posts(first:10){
        edges{
          node{
            name
            tagline
            votesCount
          }
        }
      }
    }
    """

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    r = requests.post(
        PRODUCT_HUNT_URL,
        json={"query": query},
        headers=headers
    )

    data = r.json()

    products = []

    for p in data["data"]["posts"]["edges"]:

        node = p["node"]

        products.append({
            "name": node["name"],
            "tagline": node["tagline"],
            "votes": node["votesCount"]
        })

    return products