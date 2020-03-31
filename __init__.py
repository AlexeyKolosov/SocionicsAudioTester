from nltk.util import ngrams
import speech_recognition as sr
import fasttext
import fasttext.util

def get_wordlists(LABELS):
  
    WHITE_ETHICS_WORD_LIST = []
    BLACK_ETHICS_WORD_LIST = []
    WHITE_LOGICS_WORD_LIST = []
    BLACK_LOGICS_WORD_LIST = []
    WHITE_SENSORICS_WORD_LIST = []
    BLACK_SENSORICS_WORD_LIST = []
    WHITE_INTUITION_WORD_LIST = []
    BLACK_INTUITION_WORD_LIST = []
    
    word_lists = {
        'WHITE_ETHICS_WORD_LIST': WHITE_ETHICS_WORD_LIST,
        'BLACK_ETHICS_WORD_LIST': BLACK_ETHICS_WORD_LIST,
        'WHITE_LOGICS_WORD_LIST': WHITE_LOGICS_WORD_LIST,
        'BLACK_LOGICS_WORD_LIST': BLACK_LOGICS_WORD_LIST,
        'WHITE_SENSORICS_WORD_LIST': WHITE_SENSORICS_WORD_LIST,
        'BLACK_SENSORICS_WORD_LIST': BLACK_SENSORICS_WORD_LIST,
        'WHITE_INTUITION_WORD_LIST': WHITE_INTUITION_WORD_LIST,
        'BLACK_INTUITION_WORD_LIST': BLACK_INTUITION_WORD_LIST
    }

    for label in LABELS:
        raw_phrases = []
        with open(label + "_WORD_LIST.txt", "r", encoding = 'utf-16') as file:
            raw_phrases = file.readlines()
        for phrase in raw_phrases:
            phrase = phrase.replace("\n", "").replace("\t", "").lower().strip()
            word_lists[label + '_WORD_LIST'].append(phrase)
        file.close()
    WHITE_ETHICS_WORD_LIST = list(set(WHITE_ETHICS_WORD_LIST))
    BLACK_ETHICS_WORD_LIST = list(set(BLACK_ETHICS_WORD_LIST))
    WHITE_LOGICS_WORD_LIST = list(set(WHITE_LOGICS_WORD_LIST))
    BLACK_LOGICS_WORD_LIST = list(set(BLACK_LOGICS_WORD_LIST))
    WHITE_SENSORICS_WORD_LIST = list(set(WHITE_SENSORICS_WORD_LIST))
    BLACK_SENSORICS_WORD_LIST = list(set(BLACK_SENSORICS_WORD_LIST))
    WHITE_INTUITION_WORD_LIST = list(set(WHITE_INTUITION_WORD_LIST))
    BLACK_INTUITION_WORD_LIST = list(set(BLACK_INTUITION_WORD_LIST))
    return {
        'WHITE_ETHICS_WORD_LIST': WHITE_ETHICS_WORD_LIST,
        'BLACK_ETHICS_WORD_LIST': BLACK_ETHICS_WORD_LIST,
        'WHITE_LOGICS_WORD_LIST': WHITE_LOGICS_WORD_LIST,
        'BLACK_LOGICS_WORD_LIST': BLACK_LOGICS_WORD_LIST,
        'WHITE_SENSORICS_WORD_LIST': WHITE_SENSORICS_WORD_LIST,
        'BLACK_SENSORICS_WORD_LIST': BLACK_SENSORICS_WORD_LIST,
        'WHITE_INTUITION_WORD_LIST': WHITE_INTUITION_WORD_LIST,
        'BLACK_INTUITION_WORD_LIST': BLACK_INTUITION_WORD_LIST
    }
def intersect(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

def get_all_intersections(word_lists):
    intersections = []
    for lst1, lst_name1 in zip(list(word_lists.values()), 
                               list(word_lists.keys())):
        for lst2, lst_name2 in zip(list(word_lists.values()), 
                                   list(word_lists.keys())):
            if lst_name1 != lst_name2 and len(intersect(lst1, lst2)) != 0:
                intersections.extend(intersect(lst1, lst2))
#                 print(lst_name1, lst_name2, f"intersections {intersect(lst1, lst2)}")
    return list(set(intersections))

def train_socionics_fasttext_model(LABELS,
                                   word_lists = {},
                                   save_model = True):
    intersections = get_all_intersections(word_lists) # for a while not training intersections until resolve them
    with open('socionics_fasttext_train_data.txt', 'w', encoding='utf-8') as file:
        for label, word_list in zip(LABELS, list(word_lists.values())):
            for i in range(len(word_list)):
                if word_list[i] not in intersections:
                    file.write('{} {}\n'.format(word_list[i], f"__label__{label}"))

    model = fasttext.train_supervised(verbose=3, 
                                      input='socionics_fasttext_train_data.txt',
                                      thread=1, 
                                      minCount=1, 
                                      wordNgrams=1, 
                                      lr=0.1, 
                                      epoch=80, 
                                      loss='softmax',
                                      pretrainedVectors='cc.ru.300.vec',
                                      dim=300)
    if save_model is True:
        model.save_model("socionics_fasttext_model1.ckpt")
        
def get_phrases_from_text(text):
    # for a while returns only words and 2-grams
    words_from_text = text.lower().split()
    two_grams = []
    for two_gram in list(ngrams(words_from_text, 2)):
        two_grams.append(" ".join(two_gram))
    phrases = words_from_text + two_grams
    return phrases

def predict_all(phrases,
                word_lists,
                model=None):
    WHITE_ETHICS = 0
    BLACK_ETHICS = 0
    WHITE_LOGICS = 0
    BLACK_LOGICS = 0
    WHITE_SENSORICS = 0
    BLACK_SENSORICS = 0
    WHITE_INTUITION = 0
    BLACK_INTUITION = 0                    
    for phrase in phrases:    
        # to do: implement lemmatizer before predict
        if model is not None:
            data = model.predict(phrase, k=1)
            predicted_label = str(data[0][0].replace('__label__', ''))
            if predicted_label == 'WHITE_ETHICS':
                WHITE_ETHICS = WHITE_ETHICS + 1
            elif predicted_label == 'BLACK_ETHICS':
                BLACK_ETHICS = BLACK_ETHICS + 1
            elif predicted_label == 'WHITE_LOGICS':
                WHITE_LOGICS = WHITE_LOGICS + 1
            elif predicted_label == 'BLACK_LOGICS':
                BLACK_LOGICS = BLACK_LOGICS + 1
            elif predicted_label == 'WHITE_SENSORICS':
                WHITE_SENSORICS = WHITE_SENSORICS + 1
            elif predicted_label == 'BLACK_SENSORICS':
                BLACK_SENSORICS = BLACK_SENSORICS + 1
            elif predicted_label == 'WHITE_INTUITION':
                WHITE_INTUITION = WHITE_INTUITION + 1
            elif predicted_label == 'BLACK_INTUITION':
                BLACK_INTUITION = BLACK_INTUITION + 1
        #======= manual_check_in_word_lists ==========
        if phrase in word_lists['WHITE_ETHICS_WORD_LIST']:
            WHITE_ETHICS = WHITE_ETHICS + 1
        elif phrase in word_lists['BLACK_ETHICS_WORD_LIST']:
            BLACK_ETHICS = BLACK_ETHICS + 1
        elif phrase in word_lists['WHITE_LOGICS_WORD_LIST']:
            WHITE_LOGICS = WHITE_LOGICS + 1
        elif phrase in word_lists['BLACK_LOGICS_WORD_LIST']:
            BLACK_LOGICS = BLACK_LOGICS + 1
        elif phrase in word_lists['WHITE_SENSORICS_WORD_LIST']:
            WHITE_SENSORICS = WHITE_SENSORICS + 1
        elif phrase in word_lists['BLACK_SENSORICS_WORD_LIST']:
            BLACK_SENSORICS = BLACK_SENSORICS + 1
        elif phrase in word_lists['WHITE_INTUITION_WORD_LIST']:
            WHITE_INTUITION = WHITE_INTUITION + 1
        elif phrase in word_lists['BLACK_INTUITION_WORD_LIST']:
            BLACK_INTUITION = BLACK_INTUITION + 1
    return [WHITE_ETHICS, 
            BLACK_ETHICS, 
            WHITE_LOGICS,
            BLACK_LOGICS,
            WHITE_SENSORICS,
            BLACK_SENSORICS,
            WHITE_INTUITION, 
            BLACK_INTUITION] 
            
def analyse_results(LABELS, results = [0,0,0,0,0,0,0,0]):
    # max({WHITE_ETHICS:"WHITE_ETHICS",BLACK_ETHICS:"BLACK_ETHICS"})
    # max({WHITE_LOGICS:"WHITE_LOGICS",BLACK_LOGICS:"BLACK_LOGICS"})
    # max({WHITE_SENSORICS:"WHITE_SENSORICS",BLACK_SENSORICS:"BLACK_SENSORICS"})
    # max({WHITE_INTUITION:"WHITE_INTUITION",BLACK_INTUITION:"BLACK_INTUITION"})

    values_sorted, labels_sorted = zip(*sorted(zip(results, LABELS), 
                                               reverse=True))
    print("your strong sides are ", labels_sorted[:2])
    
def record_speech_and_recognize():
    r = sr.Recognizer()
    text = ""
    with sr.Microphone() as source:
        print("SAY SOMETHING")
        audio = r.listen(source)
        print("TIME OVER, THANKS")
    try:
        text = r.recognize_google(audio, language = 'ru-RU')
        print("TEXT: ", text)
    except:
        pass
    return text
    

if __name__ == '__main__':
    LABELS = ["WHITE_ETHICS", "BLACK_ETHICS", 
              "WHITE_LOGICS","BLACK_LOGICS",
              "WHITE_SENSORICS","BLACK_SENSORICS",
              "WHITE_INTUITION","BLACK_INTUITION"]
    # text = record_speech_and_recognize()
    # phrases = get_phrases_from_text(text)
    word_lists = get_wordlists(LABELS)
    train_socionics_fasttext_model(LABELS=LABELS, word_lists=word_lists)
    # model=fasttext.load_model("socionics_fasttext_model.ckpt")
    # results = predict_all(phrases=phrases, word_lists=word_lists, model=model)
    # analyse_results(LABELS=LABELS, results=results)