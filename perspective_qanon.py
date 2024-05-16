from googleapiclient import discovery
import pandas as pd
from time import sleep

# not giving you my precious key!
API_KEY = ''

# Generates API client object dynamically based on service name and version.
service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=API_KEY)

source = pd.read_csv("Qanon_cleaned.csv")

posts = source["Post"]

# This API likes to stop completely if it encounters an error. Let's split the posts into chunks just in case

posts1 = posts[:500]
posts2 = posts[500:1000]
posts3 = posts[1000:1500]
posts4 = posts[1500:2000]
posts5 = posts[2000:2500]
posts6 = posts[2500:3000]
posts7 = posts[3000:3500]
posts8 = posts[3500:4000]
posts9 = posts[4000:4500]
posts10 = posts[4500:5000]
posts11 = posts[5000:5500]
posts12 = posts[5500:]


tox_score = []

def get_scores(list_of_posts):
    for i in list_of_posts:
        analyze_request = {
            'comment': { 'text': i },
            'requestedAttributes': {'TOXICITY': {}},
            'languages': 'en'
        }
        response = service.comments().analyze(body=analyze_request).execute()
        tox_score.append(response["attributeScores"]["TOXICITY"]["spanScores"][0]["score"]["value"])
        sleep(1.5)

get_scores(posts12)

tox_score_master = []

tox_score_master = tox_score_master + tox_score

source["Toxicity_score"] = tox_score_master

source.to_csv("Qanon_with_tox_scores.csv")

