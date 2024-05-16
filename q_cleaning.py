import re
import pandas as pd

source = pd.read_csv("Qanon_pub.CSV")

r = re.compile(">>\d\d\d\d\d\d\d\d$")
test1 = list(filter(r.match,posts))

source = source[~source["Post"].isin(test1)]

r = re.compile(">>\d\d\d\d\d\d\d$")
test1 = list(filter(r.match,posts))

source = source[~source["Post"].isin(test1)]

r = re.compile(">>\d\d\d\d\d\d$")
test1 = list(filter(r.match,posts))

source = source[~source["Post"].isin(test1)]

r = re.compile(">>\d\d\d\d\d$")
test1 = list(filter(r.match,posts))

source = source[~source["Post"].isin(test1)]

r = re.compile(">>\d\d\d\d$")
test1 = list(filter(r.match,posts))

source = source[~source["Post"].isin(test1)]

r = re.compile(">>\d\d\d$")
test1 = list(filter(r.match,posts))

source = source[~source["Post"].isin(test1)]

# indices to drop
todrop= [81, 232, 256, 310, 370, 892, 994, 1095, 1196, 1337, 1342, 1387, 1388, 1390, 1392, 1454, 1455, 1458, 1459, 1482, 1488, 1489, 1526, 1615, 1827, 1895, 1958, 1962, 1985, 2021, 2394, 2396, 2658, 2800, 3007, 3148, 3173, 3328, 3453, 3565, 3895, 3903, 3908, 4055,4086,4090, 4189, 4282, 4314, 4317, 4411, 4671, 4738, 4762, 4778, 4795, 4838, 4879, 4897, 4914, 5013, 5033, 5072, 5094, 5288, 5289, 5290, 5366, 5368, 5503, 5550, 5594, 5611, 5683, 5755, 5768, 5771, 5832, 5853, 5866, 5885, 5913, 5918, 5929, 5941, 5952, 5955, 5957, 5961, 5980, 5992, 5997, 6030, 6040, 6062, 6076, 6131]

source = source.drop(index=todrop)

source = source.iloc[:,1:]

source.to_csv("Qanon_cleaned.csv")