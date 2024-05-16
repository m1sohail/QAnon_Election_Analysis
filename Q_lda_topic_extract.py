import pandas as pd
import numpy as np
import lda
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer

qanon = pd.read_csv("Qanon_with_tox_scores.csv")

qanon["tokenized"] = [word_tokenize(i) for i in qanon["Post"]]

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

# 17537 reduced to 13418

from scipy import sparse

whoa = sparse.lil_matrix(sparse.csr_matrix(Q)[:,4119:])

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
        if word not in ["www","http","com","https","pdf","html"]:
            new_words.append(word)
    return new_words

Q_anon_topics["no_html"] = Q_anon_topics.apply(lambda row: remove_html(row['no_SW']), axis=1)

Q_anon_topics["cleaned"] = [TreebankWordDetokenizer().detokenize(i) for i in Q_anon_topics["no_html"]]

QQ = pd.DataFrame()
QQ = Q_anon_topics["cleaned"].str.split(' ', expand=True)

QQ[0] = QQ[0].str.cat(QQ[1],sep=" ")

QQ = QQ.drop(columns=1)

QQ.to_csv("Q_topics.csv", index=False, header=False)

for i in range(15):
    print('{}'.format(" ".join(QQ.iloc[i,:].dropna())))

# let's make some keywords for each topic to search through the posts

# war, government, and violence
topic_0 = ["government","control","person","freedom", "military", "antifa", "violence", "warfare", "social","intelligence"]

# Comey, Epstein, news stations
topic_1 = ["treason", "believe", "comey", "epstein", "coincidences", "cnn", "barlow", "snowden","msnbc","nyt","abc","clowns"]

# clinton and sex trafficking
topic_2 = ["republican","child","democratic","sex", "sexual","democrat","guilty","traffic","girl","children","clinton","pornography","arrest"]

# fake news controlling the narrative
topic_3 = ["control","news","people","fake","media","narrative","attack","potus","truth","power","fear","divided"]

# Comey's firing
topic_4 = ["fired","fbi","director","clinton","justice","comey","attorney","counsel","doj","former","chief"]

# Surveillance organizations / foreign security threats
topic_5 = ["fisa","fbi","russia","intel","sec","uk","nsa","hussein","cia","potus","target","intelligence","hrc","foreign","doj"]

# Mueller investigation
topic_6 = ["mueller","sessions","house","fbi", "huber","potus","doj","evidence","report","senate","investigation","barr","declas","panic"]

# Iran and Saddam Hussein funding
topic_7 = ["potus","us","relevant","control","iran","money","hussein","fund","lose","deal","hrc"]

# Red October
topic_8 = ["people","right","world", "know", "power", "public", "change", "red", "light", "truth"]

# China controlling the central bank/fb/google
topic_9 = ["bank","central","china","fb","google","access","target","national","offline","data","track","mission","expand","code"]

# Twitter and youtube URLs. 
topic_10 = ["twitter","status","youtube","realdonaldtrump","hillaryclinton","mobile","patriot","saracarterdc"]

# Virus and the election
topic_11 = ["vote", "election", "china", "biden", "covid", "voter", "death", "virus", "officials", "law", "prevent", "illegal", "pelosi", "attempt"]

# rise up patriots
topic_12 = ["god", "stand", "fight", "patriots", "people", "together", "evil", "united", "bless", "ready", "country", "america", "power", "faith"]

# watch for messages (in the news, etc)
topic_13 = ["think", "know", "potus", "news", "coincidence", "future", "anons", "watch", "boom", "important", "last", "today", "message"]

# foxnews, conspiracies
topic_14 = ["news", "trump", "politics", "qanon", "foxnews", "conspiracy", "story", "nytimes", "breitbart", "article", "theory", "thehill", "russia"]

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

import pandas as pd
test = pd.read_csv("Q_tox_by_topic.csv",header=None)
import numpy as np

import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap

cmap=LinearSegmentedColormap.from_list('rg',["g", "y", "r"], N=256)

rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

data = test[1]
plt.bar(range(len(data)),data,color=cmap(rescale(data)))
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
plt.xlabel("Topic")
plt.ylabel("Average Toxicity Score")
plt.title("Average Toxicity of Qanon Posts by Topic")
axes=plt.gca()
axes.set_ylim([0,0.41])
plt.show()

tox_by_topic.to_csv("Q_tox_by_topic.csv",header=False)
