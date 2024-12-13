# Prompt Story Generator

Prompt Story Generator is an interactive storytelling system designed to generate and evaluate creative stories using advanced prompting techniques, including [zero-shot](https://www.promptingguide.ai/techniques/zeroshot), [few-shot](https://www.promptingguide.ai/techniques/fewshot), and [reflexion prompting](https://www.promptingguide.ai/techniques/reflexion). This system allows users to select a story title and genre or provide custom prompts, leveraging a fine-tuned GPT model to create tailored stories. It evaluates generated stories using robust metrics like ROUGE, BLEU, METEOR, and coherence, providing insights into the effectiveness of different prompting strategies.


## Features

- **Story Generation**: Supports zero-shot, few-shot, and reflexion prompting techniques for diverse story outputs.
- **Evaluation Metrics**: Analyzes generated stories using ROUGE, BLEU, METEOR, and coherence metrics.
- **Visualization**: Graphically compares prompting strategies and their effectiveness.
- **Interactive Interface**: Simple web interface to select titles, genres, or provide custom prompts.

---

## Getting Started

### Prerequisites

1. Install [Anaconda](https://www.anaconda.com/)
2. Clone this repository:
   ```bash
   git clone https://github.com/srinivas-badiga/Story_Generation_and_Prompt_Evaluation.git
   ```

---

### Installing Conda

To use this project, you will need Anaconda or Miniconda installed on your system. Follow these steps:

1. Download the installer:
   - [Anaconda](https://www.anaconda.com/products/distribution#download-section) (full version with additional tools)
   - [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (lightweight version)

2. Install Conda:
   - For Windows:
     - Double-click the downloaded installer and follow the on-screen instructions.
   - For macOS/Linux:
     - Open a terminal and run the following command (replace the filename with the actual downloaded file):
       ```bash
       bash Miniconda3-latest-Linux-x86_64.sh
       ```
     - Follow the prompts to complete the installation.

3. Verify the installation:
   ```bash
   conda --version
   ```

---

### Setup Instructions

1. Navigate to the project directory:
   ```bash
   cd Story_Generation_and_Prompt_Evaluation
   ```

2. Create a virtual environment using `conda`:
   ```bash
   conda create -n test python=3.10
   conda activate test
   ```

3. Install `pip` and the required dependencies:
   ```bash
   conda install pip
   pip install -r requirements.txt
   ```

4. Download the necessary NLP model:
   ```bash
   python -m spacy download en_core_web_md
   ```

---

### Running the Application

1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and go to:
   ```
   http://127.0.0.1:5000
   ```

---

## Project Workflow

1. **User Input**: Users can select a story title and genre or provide custom prompts through the web interface.
2. **Story Generation**: The backend processes the input and generates stories using the selected prompting technique.
3. **Evaluation**: The system evaluates the generated stories using metrics such as ROUGE, BLEU, METEOR, coherence and confidence scores to measure quality and relevance.
4. **Visualization**: Results are presented through comparative graphs and scores for user review.

Additionally, the system includes two distinct modules:
- **Default Prompt Module**: Generates stories using predefined prompts.
- **Custom Prompt Module**: Allows users to craft and generate stories based on their custom inputs.
---


## Repository Structure

- **`app.py`**: Main Flask application.
- **`templates/`**: HTML templates for the web interface.
- **`static/`**: CSS and JavaScript files for styling and interactivity.
- **`data.csv`**: Dataset containing story titles and reference stories.
- **`requirements.txt`**: List of dependencies for the project.
- **`System Diagram.png`**: System Architecture.

---

## Technologies Used

- **Python 3.10**
- **Flask**
- **OpenAI GPT API**
- **spaCy**
- **Matplotlib**

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Acknowledgments

- **Dataset**: [ROCStories-MOD](https://huggingface.co/datasets/aummthaker/ROCSTORIES-MOD)
- **LLM**: Fine-tuned GPT model for storytelling.
