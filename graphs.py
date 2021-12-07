import pandas as pd
import seaborn as sns

data = pd.read_csv('advertising.csv')
df = pd.DataFrame(data)

df.head(51).plot(x="TV", y="Sales", kind="bar", xlabel="Investment on TV(in lakhs)", ylabel="Sales(in cr)")
df.head(51).plot(x="SocialMedia", y="Sales", kind="bar", xlabel="Investment on SocialMedia(in lakhs)", ylabel="Sales(in cr)")
df.head(51).plot(x="Newspaper", y="Sales", kind="bar", xlabel="Investment on Newspaper(in lakhs)", ylabel="Sales(in cr)")

sns.heatmap(df.corr())
