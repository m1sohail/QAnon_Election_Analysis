import pandas as pd
import numpy as np
import lda
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer

qanon = pd.read_csv("Trump_with_tox_scores.csv")

qanon["tokenized"] = [word_tokenize(i) for i in qanon["text"]]

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

qanon['normalized'] = qanon.apply(lambda row: lemmatize_verbs(row['tokenized']), axis=1)

def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

qanon["no_SW"] = qanon.apply(lambda row: remove_stopwords(row['normalized']), axis=1)


qanon["LDA_input"] = [TreebankWordDetokenizer().detokenize(i) for i in qanon["no_SW"]]

vec = CountVectorizer()
q_docs = qanon["LDA_input"]
Q = vec.fit_transform(q_docs)
df = pd.DataFrame(Q.toarray(), columns=vec.get_feature_names())

# lets drop the columns that start with numbers

def drop_annoying_numbers(fun):
    for i in range(10):
        fun = fun.loc[:, ~fun.columns.str.startswith(str(i))]
    return fun

df = drop_annoying_numbers(df)

# can't do much about the underscores at this point. should be enough

# 19310 reduced to 17871

from scipy import sparse

whoa = sparse.lil_matrix(sparse.csr_matrix(Q)[:,1439:])

vocab = df.columns

model = lda.LDA(n_topics=15, n_iter=3000, random_state=42069)

model.fit(whoa)

topic_word = model.topic_word_

# since I included URLs, there may be lots of garbage. let's sift through the top 30

n_top_words = 30

Q_anon_topics = []

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    Q_anon_topics.append('Topic {}: {}'.format(i, ' '.join(topic_words)))

# some stopwords left in there for some reason. Probably leftovers from urls. clean a bit.

Q_anon_topics = pd.DataFrame(Q_anon_topics)

Q_anon_topics["tokenized"] = [word_tokenize(i) for i in Q_anon_topics[0]]
Q_anon_topics["no_SW"] = Q_anon_topics.apply(lambda row: remove_stopwords(row['tokenized']), axis=1)

def remove_html(words):
    new_words = []
    for word in words:
        if word not in ["www","http","com","https","pdf","html","co"]:
            new_words.append(word)
    return new_words

Q_anon_topics["no_html"] = Q_anon_topics.apply(lambda row: remove_html(row['no_SW']), axis=1)

Q_anon_topics["cleaned"] = [TreebankWordDetokenizer().detokenize(i) for i in Q_anon_topics["no_html"]]

QQ = pd.DataFrame()
QQ = Q_anon_topics["cleaned"].str.split(' ', expand=True)

QQ[0] = QQ[0].str.cat(QQ[1],sep=" ")

QQ = QQ.drop(columns=1)

QQ.to_csv("Trump_topics.csv", index=False, header=False)

for i in range(15):
    print('{}'.format(" ".join(QQ.iloc[i,:].dropna())))

# let's make some keywords for each topic to search through the posts

# New books about Trump
topic_0 = ["trump", "president", "book", "donald", "new", "great", "mike", "america", "presidential", "foxandfriends", "american"]

# The wicked fake news
topic_1 = ["fake", "news", "media", "report", "cnn", "bad", "know", "story", "rat", "corrupt", "totally"]

# America is awesome, also god
topic_2 = ["today", "american", "great", "honor", "day", "america", "nation", "god", "families", "country", "americans", "stand", "love","bless"]

# Other countries screwing over US
topic_3 = ["china", "trade", "pay", "dollars", "make", "deal", "countries", "money", "tariffs", "price", "company", "farmers"]

# The do-nothing democrats, impeachment
topic_4 = ["democrats", "nothing", "impeachment", "president", "call", "nancy", "never", "pelosi", "democrat", "schiff", "left", "radical","hoax", "dems"]

# Sleepy Joe and the election
topic_5 = ["biden", "joe", "vote", "win", "election", "sleepy", "state", "ballot", "radical", "left", "bernie", "fraud", "republicans"]

# Fox News is awesome
topic_6 = ["house", "white", "foxnews", "interview", "congratulations", "conference", "secretary", "foxandfriends", "respect", "general", "seanhannity"]

# MAGA and key election states
topic_7 = ["great", "america", "maga", "make", "florida", "carolina", "happy", "rally", "vote", "crowd", "pennsylvania", "north", "beautiful"]

# Stay at home orders / police
topic_8 = ["state", "law", "help", "federal", "new", "york", "government", "california", "governor", "mayor", "enforcement", "order", "fund", "police", "safe", "guard", "support"]

# Crooked Hillary NO COLLUSION
topic_9 = ["hunt", "witch", "collusion", "fbi", "mueller", "russia", "hillary", "campaign", "crooked", "report", "clinton", "comey", "obama", "justice", "russian", "lie", "hoax", "investigation"]

# The economy's doing great...
topic_10 = ["economy", "great", "ever", "record", "best", "history", "tax", "jobs", "market", "republican", "stock", "high", "approval"]

# Support the military / endorsements
topic_11 = ["endorsement", "complete", "strong", "amendment", "military", "vets", "crime", "congressman", "governor", "senator", "border", "support", "fight"]

# Something something Obamagate
topic_12 = ["breitbartnews", "thank", "via", "agree", "promises", "congratulations", "nice", "obamagate","amazing", "mark", "read", "truth", "watcher", "franklin", "disgraceful"]

# Build the wall / supreme court
topic_13 = ["border", "wall", "security", "court", "immigration", "crime", "work", "mexico", "southern", "build", "supreme", "illegal"]

# Iran China North Korea
topic_14 = ["president", "meet", "korea", "north", "iran", "china", "forward", "minister", "trade", "kim","years"]

def extract_toxicity_scores(topic):
    emptylist = [[]]
    for i in range(len(qanon)):
        matches = {word for word in topic if word in qanon["LDA_input"][i]}
        if len(matches) > 2:
            emptylist.append([qanon["LDA_input"][i],qanon["Toxicity_score"][i]])
    return pd.DataFrame(emptylist).dropna()

tox_by_topic = pd.DataFrame(np.nan,index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],columns=["Tox_score"])


list_topics = [topic_0,topic_1,topic_2,topic_3,topic_4,topic_5,topic_6,topic_7,topic_8,topic_9,topic_10,topic_11,topic_12,topic_13,topic_14]

for i in range(15):
    temp = extract_toxicity_scores(list_topics[i])
    tox_by_topic["Tox_score"][i] = np.mean(temp[1])


import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap

cmap=LinearSegmentedColormap.from_list('rg',["g", "y", "r"], N=256)

rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

data = tox_by_topic["Tox_score"]
plt.bar(range(len(data)),data,color=cmap(rescale(data)))
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
plt.xlabel("Topic")
plt.ylabel("Average Toxicity Score")
plt.title("Average Toxicity of Trump Tweets by Topic")
plt.show()

tox_by_topic.to_csv("Trump_tox_by_topic.csv",header=False)
