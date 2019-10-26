import pandas as pd


if __name__=="__main__":
	data=pd.read_json("News_Category_Dataset_v2.json", lines=True)
	print("The number of unique categories in file News_Category_Dataset_v2.json is:")
	print(len(data['category'].unique()))