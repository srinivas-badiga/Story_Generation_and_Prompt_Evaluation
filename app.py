from flask import Flask, render_template, request, jsonify
import pandas as pd
import openai
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import spacy
import math
import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize

# Initialize Flask app
app = Flask(__name__)

# Load ROC Stories dataset
roc_stories_df = pd.read_csv('data.csv')
titles = roc_stories_df['storytitle'].unique().tolist()
genres = ["Comedy", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller"]

# OpenAI API configuration
openai.api_key = "*******"
fine_tuned_model_id = "ft:gpt-4o-mini-2024-07-18:cognizant::ATuTY5OP"
#fine_tuned_model_id = "gpt-4o-mini-2024-07-18"
# Load spaCy model for coherence calculation
nlp = spacy.load("en_core_web_md")

# Evaluation functions (ROUGE, BLEU, METEOR, Coherence)
def evaluate_rouge(prediction, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    score = scorer.score(reference, prediction)
    return score

def evaluate_bleu(prediction, reference):
    reference_tokens = reference.split()
    prediction_tokens = prediction.split()
    return sentence_bleu([reference_tokens], prediction_tokens)
'''
def evaluate_meteor(prediction, reference):
    
    return meteor_score([reference], prediction)'''


def evaluate_meteor(prediction, reference):
    # Tokenize both the prediction and reference
    prediction_tokens = word_tokenize(prediction)  # Tokenize the prediction
    reference_tokens = word_tokenize(reference)    # Tokenize the reference
    return meteor_score([reference_tokens], prediction_tokens)


def evaluate_coherence(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    
    # Calculate cosine similarity between adjacent sentences for coherence
    similarities = []
    for i in range(len(sentences) - 1):
        vec1 = sentences[i].vector
        vec2 = sentences[i+1].vector
        similarity = vec1 @ vec2 / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        similarities.append(similarity)
    
    # Average coherence score
    coherence_score = np.mean(similarities)
    
    # If coherence is NaN, return 0.0
    if math.isnan(coherence_score):
        return 0.0
    
    return coherence_score

def evaluate_confidence(rouge_scores, bleu_score, meteor_score, coherence):
    # Combine metrics into a single confidence score (weighted average)
    # You can adjust the weights based on importance
    rouge_weight = 0.25
    bleu_weight = 0.25
    meteor_weight = 0.25
    coherence_weight = 0.25
    
    confidence_score = (
        rouge_weight * (rouge_scores['rouge1'].fmeasure + rouge_scores['rouge2'].fmeasure + rouge_scores['rougeL'].fmeasure) / 3 +
        bleu_weight * bleu_score +
        meteor_weight * meteor_score +
        coherence_weight * coherence
    )
    
    # Normalize the confidence score to be between 0 and 1
    max_possible_score = 1.0
    return min(confidence_score / max_possible_score, 1.0)



# Generate Story using OpenAI API
def generate_stories(prompt, model_id=fine_tuned_model_id):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "Write a 150-word detailed story about a strong central character embarking on a journey filled with mystery and discovery, with vivid descriptions, logical progression, and a clear beginning, middle, and end, emphasizing exploration, wonder, and rich character development in fluent, original language."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content'].strip()

def generate_stories_with_reflexion(prompt, model_id=fine_tuned_model_id):
    # Initial story generation
    initial_response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "Write a 150-word detailed story about a strong central character embarking on a journey filled with mystery and discovery, with vivid descriptions, logical progression, and a clear beginning, middle, and end, emphasizing exploration, wonder, and rich character development in fluent, original language."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
    )
    
    initial_story = initial_response['choices'][0]['message']['content'].strip()
    
    # Reflexion step: Ask the model to evaluate and revise the story
    reflection_response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "Evaluate the following story for coherence, vivid descriptions, logical progression, and adherence to the theme of mystery and discovery. Identify areas where the story could be improved, and then provide a revised version with these improvements applied. And return the 150-word detailed story based on this evaluation. Do not include anything but the new generated story in the response"},
            {"role": "user", "content": initial_story}
        ],
        max_tokens=500,
        temperature=0.7,
    )
    
    revised_story = reflection_response['choices'][0]['message']['content'].strip()
    
    return revised_story

# Plotting function for evaluation results
def plot_metrics(results):
    techniques = list(results.keys())
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'METEOR', 'Coherence']
    
    # Collecting scores
    scores = {metric: [] for metric in metrics}
    for technique, result in results.items():
        rouge = result['rouge_scores']
        bleu = result['bleu_score']
        meteor = result['meteor_score']
        coherence = result['coherence']
        scores['ROUGE-1'].append(round(rouge['rouge1'].fmeasure, 6))
        scores['ROUGE-2'].append(round(rouge['rouge2'].fmeasure, 6))
        scores['ROUGE-L'].append(round(rouge['rougeL'].fmeasure, 6))
        scores['BLEU'].append(round(bleu, 6))
        scores['METEOR'].append(round(meteor, 6))
        scores['Coherence'].append(round(coherence, 6))

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 10))
    bar_width = 0.12
    colors = ['#FF0000', '#87CEEB', '#800080', '#FFA500', '#FFFF00', '#90EE90']
    
    for i, metric in enumerate(metrics):
        y_positions = np.arange(len(techniques)) + i * bar_width
        ax.barh(y_positions, scores[metric], height=bar_width, label=metric, color=colors[i % len(colors)])
    
    ax.set_yticks(np.arange(len(techniques)) + bar_width)
    ax.set_yticklabels(techniques)
    ax.set_xlabel('Scores', loc='center')
    ax.set_title('Evaluation Scores for All Techniques', loc='center')
    ax.legend()

    # Convert plot to image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.tight_layout()  # Automatically adjusts padding and spacing

    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

# Route to handle the form and display results
@app.route('/')
def index():
    return render_template('index.html')  # Default route now points to index page

# Route for 'Overview'
@app.route('/overview')
def overview():
    # Add your project details, professor info, team members, etc.
    return render_template('overview.html')

# Route for 'Generate Story with Default Prompts'
@app.route('/generate-story', methods=['GET', 'POST'])
def generate_story():
    if request.method == 'POST':
        story_title = request.form['story_title']
        genre = request.form['genre']

        # Fetch the story text from the dataset
        story_row = roc_stories_df.loc[roc_stories_df['storytitle'] == story_title]
        reference_story = story_row['story'].values[0] if not story_row.empty else "Original story not found."

        # Define prompt types
        prompts = {
            "zero-shot": f"Write a story based on the title '{story_title}' in a {genre} style.",
            "few-shot": f"Now, write a story based on the title '{story_title}' in a {genre} style. Here is an example story based on the title 'Jane's New Job':\n\nJane had recently gotten a new job. She was nervous about her first day of work. On the first day of work, Jane overslept. Jane arrived at work an hour late. Jane did not make a good impression at her new job.\n\n",
            "reflexion": f"Write a story based on the title  '{story_title}' in a {genre} style. Reflect on whether the story covers the main points effectively and revise if needed."
        }

        results = {}
        best_technique = None
        best_confidence = -1

        # Generate stories and evaluate each prompting technique
        for prompt_type, prompt in prompts.items():
            if prompt_type=="reflexion":
                summary = generate_stories_with_reflexion(prompt)
            else:
                summary = generate_stories(prompt)
            rouge_scores = evaluate_rouge(summary, reference_story)
            bleu_score = evaluate_bleu(summary, reference_story)
            meteor_score_val = evaluate_meteor(summary, reference_story)
            coherence = evaluate_coherence(summary)
            # Calculate the confidence score
            confidence_score = evaluate_confidence(rouge_scores, bleu_score, meteor_score_val, coherence)


            # Store results
            results[prompt_type] = {
                "summary": summary,
                "rouge_scores": rouge_scores,
                "bleu_score": bleu_score,
                "meteor_score": meteor_score_val,
                "coherence": coherence,
                "confidence_score": round(confidence_score, 3)
            }

            # Determine the best technique based on BLEU score
            if confidence_score > best_confidence:
                best_confidence = confidence_score
                best_technique = prompt_type

        # Plot the metrics for all techniques
        plot_image = plot_metrics(results)

        return render_template(
            'result.html',
            results=results,
            plot_image=plot_image,
            best_technique=best_technique  # Pass best technique to template
        )

    return render_template('generate_story.html', titles=titles, genres=genres)

# Route for 'Generate Story with Custom Prompts'
@app.route('/generate-story-custom', methods=['GET', 'POST'])
def generate_story_custom():
    if request.method == 'POST':
        story_title = request.form['story_title']
        genre = request.form['genre']
        zero_shot_prompt = request.form['zero_shot_prompt']
        few_shot_prompt = request.form['few_shot_prompt']
        reflexion_prompt = request.form['reflexion_prompt']


        # Fetch the story text from the dataset
        story_row = roc_stories_df.loc[roc_stories_df['storytitle'] == story_title]
        reference_story = story_row['story'].values[0] if not story_row.empty else "Original story not found."

        prompts = {
            "zero-shot": f"Write a {genre} story based on the title '{story_title}'. {zero_shot_prompt}",
            "few-shot": f"Write a {genre} story based on the title '{story_title}'. {few_shot_prompt}",
            "reflexion": f"Write a {genre} story based on the title  '{story_title}'. {reflexion_prompt}"
        }
        '''
        prompts = {
            "zero-shot": zero_shot_prompt,
            "few-shot": few_shot_prompt,
            "reflexion": reflexion_prompt
        }
        '''

        

        results = {}
        best_technique = None
        best_confidence = -1

        # Generate stories and evaluate each prompting technique
        for prompt_type, prompt in prompts.items():
            if prompt_type=="reflexion":
                summary = generate_stories_with_reflexion(prompt)
            else:
                summary = generate_stories(prompt)
            rouge_scores = evaluate_rouge(summary, reference_story)
            bleu_score = evaluate_bleu(summary, reference_story)
            meteor_score_val = evaluate_meteor(summary, reference_story)
            coherence = evaluate_coherence(summary)
            # Calculate the confidence score
            confidence_score = evaluate_confidence(rouge_scores, bleu_score, meteor_score_val, coherence)


            # Store results
            results[prompt_type] = {
                "summary": summary,
                "rouge_scores": rouge_scores,
                "bleu_score": bleu_score,
                "meteor_score": meteor_score_val,
                "coherence": coherence,
                "confidence_score": round(confidence_score, 3)
            }

            # Determine the best technique based on BLEU score
            if confidence_score > best_confidence:
                best_confidence = confidence_score
                best_technique = prompt_type


        # Plot the metrics for all techniques
        plot_image = plot_metrics(results)

        return render_template(
            'result.html',
            results=results,
            plot_image=plot_image,
            best_technique=best_technique  # Pass best technique to template
        )

    return render_template('generate_story_custom.html', titles=titles, genres=genres)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
