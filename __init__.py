from nltk.util import ngrams
import speech_recognition as sr
import fasttext

def get_wordlists(LABELS):
    LABELS = ["WHITE_ETHICS", "BLACK_ETHICS", 
              "WHITE_LOGICS","BLACK_LOGICS",
              "WHITE_SENSORICS","BLACK_SENSORICS",
              "WHITE_INTUITION","BLACK_INTUITION"]

    WHITE_ETHICS_WORD_LIST = []
    BLACK_ETHICS_WORD_LIST = []
    WHITE_LOGICS_WORD_LIST = []
    BLACK_LOGICS_WORD_LIST = []
    WHITE_SENSORICS_WORD_LIST = []
    BLACK_SENSORICS_WORD_LIST = []
    WHITE_INTUITION_WORD_LIST = []
    BLACK_INTUITION_WORD_LIST = []

    for label, word_list in zip(LABELS, [WHITE_ETHICS_WORD_LIST, 
                                         BLACK_ETHICS_WORD_LIST, 
                                         WHITE_LOGICS_WORD_LIST, 
                                         BLACK_LOGICS_WORD_LIST, 
                                         WHITE_SENSORICS_WORD_LIST, 
                                         BLACK_SENSORICS_WORD_LIST, 
                                         WHITE_INTUITION_WORD_LIST, 
                                         BLACK_INTUITION_WORD_LIST]):
        with open(label + "_WORD_LIST.txt", "r", encoding = 'utf-16') as file:
            word_list = file.readlines()
        for i in range(len(word_list)):
            word_list[i] = word_list[i].replace("\n", "").lower()
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


def train_socionics_fasttext_model(save_model = True, 
                                   LABELS,
                                   word_lists = {}):
    with open('socionics_fasttext_train_data.txt', 'w', encoding='utf-16') as file:
        for label, word_list in zip(LABELS, list(word_lists.values())):
            for i in range(len(word_list):
                file.write('{} {}\n'.format(word_list[i], f"__{label}__"))
    
    model = fasttext.train_supervised(verbose=0, 
                                      input='socionics_fasttext_train_data.txt',
                                      thread=1, 
                                      minCount=1, 
                                      wordNgrams=1, 
                                      lr=0.1, 
                                      epoch=80, 
                                      loss='softmax')
    if save_model is True:
        model.save_model("socionics_fasttext_model.ckpt")
        
def get_phrases_from_text(text):
    # for a while returns only words and 2-grams
    words_from_text = text.lower().split()
    two_grams = []
    for two_gram in list(ngrams(words_from_text, 2)):
        two_grams.append(" ".join(two_gram))
    phrases = words_from_text + two_grams
    return phrases

def predict_all(model=fasttext.load_model("socionics_fasttext_model.ckpt"), 
                phrases,
                word_lists):
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
          "WHITE_INTUITION","BLACK_INTUITION"]s
    text = record_speech_and_recognize()
    phrases = get_phrases_from_text(text)
    word_lists = get_wordlists(LABELS)
    train_socionics_fasttext_model(LABELS=LABELS, word_lists=word_lists)
    results = predict_all(phrases=phrases, word_lists=word_lists)
    analyse_results(LABELS=LABELS, results=results)