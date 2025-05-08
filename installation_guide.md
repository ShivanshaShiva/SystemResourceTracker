# Installation Guide

## Required Dependencies

To run this application locally, you'll need to install the following Python packages:

```
streamlit>=1.31.0
pandas>=2.1.0
numpy>=1.26.0
plotly>=5.18.0
scikit-learn>=1.3.0
nltk>=3.8.0
trafilatura>=1.6.0
gitpython>=3.1.30
tiktoken>=0.5.0
```

## Local Setup Instructions

1. **Clone or download the repository**
   - Either download the ZIP file from Replit
   - Or clone from GitHub if the repository has been pushed there

2. **Create a virtual environment (recommended)**
   ```bash
   # Create a virtual environment
   python -m venv venv
   
   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Install all required packages
   pip install streamlit pandas numpy plotly scikit-learn nltk trafilatura gitpython tiktoken
   
   # Or if you created a requirements.txt file:
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```
   The application will start on http://localhost:8501 by default

## Optional: API Keys

Some features (Code Generation) require API keys:
- OpenAI API key: Get from https://platform.openai.com/api-keys
- Anthropic API key: Get from https://console.anthropic.com/

## File Structure

- `app.py`: Main Streamlit application
- `utils/`: Directory containing utility modules
  - `sanskrit_nlp.py`: Sanskrit NLP processing
  - `instruction_learner.py`: General instruction learning
  - `data_processor.py`: Data processing utilities
  - `model_trainer.py`: Custom model training
  - `code_generator.py`: AI code generation
  - `repo_processor.py`: GitHub repository processing
  - `visualizer.py`: Data visualization utilities

## Customization

You can customize the application by:
1. Modifying the existing modules in the `utils/` directory
2. Adding new modules for additional functionality
3. Editing `app.py` to update the UI or add new features

To add new instruction learning patterns:
1. Open `utils/instruction_learner.py`
2. Add new pattern recognition in the relevant methods:
   - `_learn_definition()`
   - `_learn_equation()`
   - `_learn_translation()`
   - Or add entirely new learning methods

## Persistence

The learned instructions are saved to `saved_instructions.json` in the root directory. 
This allows learned knowledge to persist between application restarts.