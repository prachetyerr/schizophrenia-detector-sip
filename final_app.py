import pandas as pd
import streamlit as st

#data reading
data = pd.read_csv('Data/Suicide_Detection.csv')

#data writing (first 15 rows, out of 2,32,000)
st.markdown("""
<style>
.big-font {
    font-size:40px !important;
}
.result-font{
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Schizophrenia Detector</p>', unsafe_allow_html=True)

def training_model():
    #training the model:

    #train-test split
    from sklearn.model_selection import train_test_split
    X = data.drop(labels=['class', 'Unnamed: 0' ], axis=1)
    y = data['class']
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True)

    #vectorisation of text
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(x_train['text'])

    #transformation of text
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    #st.write(X_train_tfidf.shape)

    #naive bayes model
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    #example - will be commented out
    docs_new = ['I wanna kill myself', 'I love KFC']
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    #for doc, category in zip(docs_new, predicted):
        #st.write(f'{doc} -> {category}')

    #first iteration of the model with naive bayes
    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    text_clf.fit(x_train['text'], y_train)

    #checking accuracy and metrics - will be commented out as well
    #st.write(f"accuracy:-> {text_clf.score(x_test['text'],y_test)}")

    #second iteration of the model with SGD pipeline
    from sklearn.linear_model import SGDClassifier
    text_clf2 = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                            alpha=.0001, random_state=42,
                            max_iter=5, tol=None)),
    ])

    text_clf2.fit(x_train['text'], y_train)

    docs_test = x_test['text']
    predicted = text_clf2.predict(docs_test)
    #st.write(f"accuracy:-> {text_clf2.score(x_test['text'],y_test)}")

    #choosing second model (SGD) as our better model
    best_model = text_clf2
    return best_model

#saving the model and loading it (to link it on to the frontend)
import joblib

filename = 'finalized_model.sav'
joblib.dump(training_model(), filename)
loaded_model=joblib.load(filename)

#some sample data for testing
post_data=["i have no reason to live anymore",
           "I passed my driver's permit test today",
           "I feel like I'm at my end.",
           "I'm tired of convincing myself that i want to be alive every day.",
           "We first got together freshman year",]

#taking input from user 
input = st.text_input('Enter the comment')
st.write('You have entered:', input)
input_data=[input]
if st.button('Predict', type='primary'):
    final_data=input_data
else:
    final_data=post_data


#logic to display
hits=0
res=loaded_model.predict(final_data)
for i in range(len(res)):
    if res[i]=="non-suicide":
        res[i]=0
    else:
        res[i]=1
res=list(map(int,res))
ratio_res=sum(res)/len(res)
if ratio_res>0.4:
    st.markdown('<p class="result-font">Result: Person appears to be schizophrenic</p>', unsafe_allow_html=True)
else:
    st.markdown('<p class="result-font">Result: Person appears to be normal</p>', unsafe_allow_html=True)

st.write("A few examples:")
st.write(data.head(15))