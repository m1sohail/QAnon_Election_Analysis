from googleapiclient import discovery
import pandas as pd
from time import sleep

# secret key
API_KEY = ''

# Generates API client object dynamically based on service name and version.
service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=API_KEY)

source = pd.read_csv("Trump_tweets_no_RT.csv")

posts = source["text"]

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

get_scores(posts)

source["Toxicity_score"] = tox_score

source.to_csv("Trump_with_tox_scores.csv")

