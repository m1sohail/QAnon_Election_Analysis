from selenium import webdriver
import pandas as pd
import time

browser=webdriver.Firefox(executable_path=r'C:\Users\Administrator\Documents\geckodriver.exe')
browser.get("https://qanon.pub/")

date = browser.find_elements_by_class_name("time")
post = browser.find_elements_by_class_name("text")

# Put all this garbage into a dataframe

df = pd.DataFrame()

Date = []
Post = []

for i in date:
    Date.append(i.text)

for i in post:
    Post.append(i.text)

df["Date"] = Date
df["Post"] = Post

# Bunch of \n everywhere. Let's replace with spaces.

new = []

for i in df["Post"]:
    new.append(i.replace("\n"," "))

df = df.drop(columns="Post")

df["Post"] = new

# Let's drop the posts that don't have any text (contained pictures)

dropped_blanks = df[df["Post"] != '']

# Write to xlsx instead of CSV to avoid UTF encoding errors;
# Don't like â€™ everywhere there's supposed to be an apostrophe

dropped_blanks.to_excel("Qanon_pub.xlsx")
