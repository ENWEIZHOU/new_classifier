#In a terminal
#pip install bert_serving-server
#pip install bert_serving-client
# bert-serving-start -model_dir D:\PTA\NEW\bert\cased_L-12_H-768_A-12 -num_worker=2     ##change the directory to your model directory


#In .py file
# from bert_serving.client import BertClient
# bc=BertClient(ip='localhost')
# a=bc.encode(['Hello China'])
# print(a)

#In a terminal
#bert-serving-terminate -port 5555