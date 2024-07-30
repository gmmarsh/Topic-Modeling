from flask import Flask, redirect, url_for
from flask import request
from gensim import corpora, models
import pyLDAvis.gensim_models
import spacy

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Load the docs.txt file
        with open('docs.txt', 'r') as f:
            texts = f.readlines()

        # define the docs.txt file
        texts = [doc.split() for doc in texts]
        
        # spaCy model
        nlp = spacy.load('en_core_web_sm')

        # process the texts as a stream
        docs = list(nlp.pipe(' '.join(doc) for doc in texts))

        # stop words, lemmatization, punctuation, numbers
        texts = [[token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num] for doc in docs]

        # lowercase
        texts = [[word.lower() for word in doc] for doc in texts]

        # dictionary
        dictionary = corpora.Dictionary(texts)

        # corpus
        corpus = [dictionary.doc2bow(text) for text in texts]

        # coherence score for different number of topics
        coherence_scores = []
        for num_topics in range(2, 50, 2):
            lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
            coherence_model = models.CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_score = coherence_model.get_coherence()
            coherence_scores.append(coherence_score)

        # Find the optimal number of topics
        optimal_num_topics = range(2, 50)[coherence_scores.index(max(coherence_scores))]

        # Train the LDA model
        lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=optimal_num_topics)

        # Visualize the topics
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

        # Save the visualization to a file
        pyLDAvis.save_html(vis, 'lda.html')

        # coherence score to HTML
        with open('coherence.html', 'w') as f:
            f.write('<h1>Coherence Scores</h1>')
            f.write('<table>')
            f.write('<tr><th>Number of Topics</th><th>Coherence Score</th></tr>')
            for num_topics, coherence_score in enumerate(coherence_scores, start=2):
                f.write(f'<tr><td>{num_topics}</td><td>{coherence_score}</td></tr>')
            f.write('</table>')

        return redirect(url_for('visualization'))

    return '''
        <form method="POST">
            <button type="submit">Load Data</button>
        </form>
    '''

@app.route('/visualization')
def visualization():
    try:
        with open('lda.html', 'r') as f:
            lda_html = f.read()
        with open('coherence.html', 'r') as f:
            coherence_html = f.read()
        header = '<h1>Topic Modeling of Amazon Book Reviews</h1>'
        return header + lda_html + coherence_html
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)