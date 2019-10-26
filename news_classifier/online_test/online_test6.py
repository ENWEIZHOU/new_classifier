from gensim import models


if __name__=="__main__":
	print("Loading the GoogleNews-vectors-negative300.bin...")
	model = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
	cos_distance = model.distance("man", "woman")
	print("The cosine distance between the vector representations of 'man' and 'woman' is: ")
	print(cos_distance)
	print("---------------------------------------------------------------------------------------------------")
	print("")
