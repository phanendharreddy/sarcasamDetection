from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from string import punctuation
from nltk.corpus import stopwords
import pandas as pd
from emoji import UNICODE_EMOJI

main = tkinter.Tk()
main.title("EXTENSION OF THE LEXICON ALGORITHM FOR SARCASM DETECTION") #designing main screen
main.geometry("1300x1200")

sid = SentimentIntensityAnalyzer()

global filename
global dataset
global process
global sarcastic
global sentiment

def checkSarcasm(sentence):
    pos = []
    neg = []
    neu = []
    arr = sentence.split(' ')
    for i in range(len(arr)):
        word = arr[i].strip()
        if word == 'smilingfacewithhearteyes':
            word = 'excellent'
        if word == 'loudlycryingface':
            word = 'bad'
        if word == 'winkingfacewithtongue':
            word = 'happy'    
        if (sid.polarity_scores(word)['compound']) >= 0.1:
            pos.append(word)
        elif (sid.polarity_scores(word)['compound']) <= -0.1:
            neg.append(word)
        else:
            neu.append(word)
    return pos,neg,neu    

def clean_doc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens) #here upto for word based
    return tokens

def upload():
    global filename
    global dataset
    dataset = []
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    train = pd.read_csv(filename,encoding='utf8',sep='\t')
    count = 0
    for i in range(len(train)):
        tweet = train.get_value(i,0,takeable = True)
        print(tweet)
        if str(tweet) != 'nan':
            tweet = tweet.lower()
        icon = train.get_value(i,1,takeable = True)
        if str(icon) != 'nan':
            icon = UNICODE_EMOJI[icon.strip()]
            icon = ''.join(re.sub('[^A-Za-z\s]+', '', icon))
            icon = icon.lower()
        else:
            icon = ''
        msg = ''
        if str(tweet) != 'nan':
            arr = tweet.split(" ")
            for k in range(len(arr)):
                word = arr[k].strip()
                if len(word) > 2:
                    msg+=word+" "
        textdata = msg.strip()+" "+icon
        #print(textdata)
        dataset.append(textdata)
       
    text.insert(END,'Total tweets found in dataset is : '+str(len(dataset)))

def Preprocessing():
    text.delete('1.0', END)
    global process
    process = []
    text.insert(END,'Messages after preprocessing and removing stopwords\n')
    text.insert(END,'====================================================================================\n')

    for i in range(len(dataset)):
        sentence = dataset[i]
        sentence = sentence.lower()
        sentence = clean_doc(sentence)
        text.insert(END,sentence+'\n')
        process.append(sentence)
                
def firstAlgorithm():
    text.delete('1.0', END)
    global sarcastic
    sarcastic = []
    for i in range(len(process)):
        sentence = process[i]
        if sentence == 'smilingfacewithhearteyes':
            sentence = 'excellent'
        if sentence == 'loudlycryingface':
            sentence = 'bad'
        if sentence == 'winkingfacewithtongue':
            sentence = 'happy'   
        sentiment_dict = sid.polarity_scores(sentence)
        negative_polarity = sentiment_dict['neg']
        positive_polarity = sentiment_dict['pos']
        neutral_polarity = sentiment_dict['neu']
        compound = sentiment_dict['compound']
        result = ''
        if compound >= 0.1 :
            result = 'Positive' 
        elif compound <= -0.1:
            result = 'Negative' 
        else :
            result = 'Neutral'
        if result =='Positive' or result == 'Neutral':
            pos,neg,neu = checkSarcasm(sentence)
            if len(neg) > 0:
                sarcastic.append("Sarcastic")
            else:
                sarcastic.append("Non Sarcastic")
        else:
            sarcastic.append("Non Sarcastic")
        text.insert(END,'Tweets : '+dataset[i]+"\n")
        text.insert(END,'Positive Polarity : '+str(positive_polarity)+"\n")
        text.insert(END,'Negative Polarity : '+str(negative_polarity)+"\n")
        text.insert(END,'Neutral Polarity  : '+str(neutral_polarity)+"\n")
        text.insert(END,'Result : '+sarcastic[i]+"\n")
        text.insert(END,'====================================================================================\n')
            
def secondAlgorithm():
    global sentiment
    sentiment = []
    text.delete('1.0', END)
    for i in range(len(process)):
        sentence = process[i]
        if sentence == 'smilingfacewithhearteyes':
            sentence = 'excellent'
        if sentence == 'loudlycryingface':
            sentence = 'bad'
        if sentence == 'winkingfacewithtongue':
            sentence = 'happy'   
        sentiment_dict = sid.polarity_scores(sentence)
        negative_polarity = sentiment_dict['neg']
        positive_polarity = sentiment_dict['pos']
        neutral_polarity = sentiment_dict['neu']
        compound = sentiment_dict['compound']
        result = ''
        if compound >= 0.1 :
            result = 'Positive'
            sentiment.append(result)
        elif compound <= -0.1:
            result = 'Negative'
            sentiment.append(result)
        else :
            result = 'Neutral'
            sentiment.append(result)
        sar = ''    
        if result =='Positive' or result == 'Neutral':
            pos,neg,neu = checkSarcasm(sentence)
            if len(neg) > 0:
                sar = "Sarcastic"
            else:
                sar = "Non Sarcastic"
        else:
            sar = "Non Sarcastic"
            
        text.insert(END,'Tweets : '+dataset[i]+"\n")
        text.insert(END,'Positive Polarity : '+str(positive_polarity)+"\n")
        text.insert(END,'Negative Polarity : '+str(negative_polarity)+"\n")
        text.insert(END,'Neutral Polarity  : '+str(neutral_polarity)+"\n")
        text.insert(END,'Result : '+sar+"\n")
        text.insert(END,'Sentiment Prediction : '+result+'\n')
        text.insert(END,'====================================================================================\n')
    

def sarcasticGraph():
    sar = 0
    non_sar = 0
    for i in range(len(sarcastic)):
        if sarcastic[i] == "Sarcastic":
            sar = sar + 1
        if sarcastic[i] == "Non Sarcastic":
            non_sar = non_sar + 1
    height = [sar,non_sar]
    bars = ('Sarcastic','Non Sarcastic')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def sentimentGraph():
  label_X = []
  category_X = []
  pos = 0
  neg = 0
  neu = 0
  for i in range(len(sentiment)):
      if sentiment[i] == 'Positive':
          pos = pos + 1
      if sentiment[i] == 'Negative':
          neg = neg + 1
      if sentiment[i] == 'Neutral':
          neu = neu + 1    
  label_X.append('Positive')
  label_X.append('Negative')
  label_X.append('Neutral')
  category_X.append(pos)
  category_X.append(neg)
  category_X.append(neu)

  plt.pie(category_X,labels=label_X,autopct='%1.1f%%')
  plt.title('Sentiment Graph')
  plt.axis('equal')
  plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='EXTENSION OF THE LEXICON ALGORITHM FOR SARCASM DETECTION')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=100)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Social Network Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

preButton = Button(main, text="Preprocess Dataset", command=Preprocessing)
preButton.place(x=50,y=150)
preButton.config(font=font1) 

firstButton = Button(main, text="Run First System Lexicon + Polarity Computation", command=firstAlgorithm)
firstButton.place(x=50,y=200)
firstButton.config(font=font1) 

secondButton = Button(main, text="Second System Lexicon + Sentiment Prediction", command=secondAlgorithm)
secondButton.place(x=50,y=250)
secondButton.config(font=font1) 

graphButton = Button(main, text="Sentiments Graph", command=sentimentGraph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)

gButton = Button(main, text="Sarcastic Graph", command=sarcasticGraph)
gButton.place(x=50,y=350)
gButton.config(font=font1)


main.config(bg='OliveDrab2')
main.mainloop()
