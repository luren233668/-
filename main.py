import jieba
messages = jieba.cut("万里长城是中国古代劳动人民血汗的结晶和中国古代文化的象征和中华民族的骄傲")


from gensim.models.keyedvectors import KeyedVectors
w2v_model = KeyedVectors.load_word2vec_format("sgns.weibo.bigram-char.bz2", binary=False,unicode_errors='ignore')

for message in messages:
    print(w2v_model[message])


