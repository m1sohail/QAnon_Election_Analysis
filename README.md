# Analysis on QAnon Posts and Trump Tweets

## Project Background

QAnon was first seen in October 2017, when an anonymous user put a series of posts on the message board 4chan signed off as "Q". This user claimed to have a "Q Clearance" level of US security clearance (Wendling, 2021).  Eventually, these messages gained notoriety, and became known as "Q drops" or "breadcrumbs", and were often written in cryptic language peppered with slogans, pledges, and pro-Trump themes (Wendling, 2021). Present day, these messages have evolved into a wide-ranging theory that claims that President Trump is waging a secret war against elite Satan-worshipping pedophiles, who are powerful names in government, business and media (Wendling, 2021). While this initially started off as a fringe movement, it has recently gained much momentum. This is evidenced with a December poll by NPR and Ipsos that showcases 17 percent of Americans believing the core falsehood of QAnon. Recently, this movement has also been attributed to spreading falsehood around the Covid-19 pandemic, the Black Lives Matter movement, and the 2020 American presidential election. 

In order to understand this phenomenon in greater detail, this project will analyze “breadcrumbs” left by Q, as well as tweets from Donald Trump in the same respective time window. In order to do this, I will employ four types of analyses. These include :
1. Toxicity Analysis
2. Document similarity 
3. Sentiment analysis
4. Latent Dirichlet Allocation (LDA)
5. Entity Extraction

## Data Sources and Pre-Processing

For this project, we had two primary data needs, posts from the QAnon leader, and tweets from Donald Trump. Their retrieval, and applied pre-processing are discussed below:

## QAnon Posts

In order to retrieve QAnon posts, https://QAnon.pub/ was leveraged. This website has all of Q’s posts aggregated from 8kun, which is the successor of 4chan and 8chan, where QAnon posts initially appeared. In order to scrape data from this website, Selenium was used. Our team initially faced problems with the automatic scroll down feature of Selenium, as it was unresponsive towards this particular website. However, this was quickly solved by holding the “End” key for the remainder of the scraping period. Once we had our base dataset, any posts with the format “>>1234567” were extracted. This format signifies the involvement of  images, and to honor the text based scope of this project, we did not see these posts as fitting. In addition to this, our team wanted to ensure consistency of posts. In order to do this, some posts were removed manually; for example, any post containing only a single word was dropped. With all pre-processing complete, our team had a dataset of roughly 6000 QAnon posts.

## Trump Tweets

Trump tweets were retrieved from https://www.thetrumparchive.com/. This data was already available in .CSV format. Therefore, scraping was not needed. For the purpose of this project, Trump data was limited to the same time period as all available QAnon posts: October 2017 to December 2020. In addition to this, retweets were eliminated from the subset to account for the computationally expensive nature of this project. With all pre-processing complete our team had a dataset of roughly 14,000 tweets from Donald Trump.

## Findings

<img width="1167" alt="Screenshot 2024-03-20 at 11 34 24 AM" src="https://github.com/m1sohail/QAnon_Analysis/assets/77760774/8c0652c8-8e38-4086-8681-1e7f8ed652cd">

<img width="1164" alt="Screenshot 2024-03-20 at 11 34 37 AM" src="https://github.com/m1sohail/QAnon_Analysis/assets/77760774/cc93d115-90e3-47ba-b7b6-bf7260ff1e43">

<img width="1167" alt="Screenshot 2024-03-20 at 11 34 52 AM" src="https://github.com/m1sohail/QAnon_Analysis/assets/77760774/d6491fa8-1bf3-467d-b4a2-e6e4efa37461">

<img width="1165" alt="Screenshot 2024-03-20 at 11 35 05 AM" src="https://github.com/m1sohail/QAnon_Analysis/assets/77760774/17d7b436-ff5b-42c2-9878-ef7e55ba9e85">

<img width="1162" alt="Screenshot 2024-03-20 at 11 35 18 AM" src="https://github.com/m1sohail/QAnon_Analysis/assets/77760774/95433298-fb7d-46d8-9198-db240ad4eaa0">

<img width="1165" alt="Screenshot 2024-03-20 at 11 35 34 AM" src="https://github.com/m1sohail/QAnon_Analysis/assets/77760774/3754d14a-0111-460c-aa83-df8ba88c7295">










