import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from utils.sanskrit_nlp import SanskritNLP
from utils.data_processor import DataProcessor
from utils.visualizer import Visualizer
from utils.model_trainer import ModelTrainer
from utils.code_generator import CodeGenerator

# Set page configuration
st.set_page_config(
    page_title="AI Assistant & Sanskrit NLP Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables if not present
if 'sanskrit_text' not in st.session_state:
    st.session_state.sanskrit_text = ""
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = None
if 'nlp_results' not in st.session_state:
    st.session_state.nlp_results = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = None
if 'code_history' not in st.session_state:
    st.session_state.code_history = []
if 'repo_processor' not in st.session_state:
    st.session_state.repo_processor = None
if 'repos_data' not in st.session_state:
    st.session_state.repos_data = {}
if 'code_snippets' not in st.session_state:
    st.session_state.code_snippets = []
if 'similar_code' not in st.session_state:
    st.session_state.similar_code = []

# Create instances of utility classes
sanskrit_nlp = SanskritNLP()
data_processor = DataProcessor()
visualizer = Visualizer()
model_trainer = ModelTrainer()
code_generator = CodeGenerator()

# Initialize repo processor if not already done
if st.session_state.repo_processor is None:
    from utils.repo_processor import RepoProcessor
    st.session_state.repo_processor = RepoProcessor()
    
    # Add default repositories information to session state if not present
    if 'repos_data' not in st.session_state or not st.session_state.repos_data:
        st.session_state.repos_data = {
            "qiskit": {
                "url": "https://github.com/Qiskit/qiskit",
                "language": "Python",
                "processed": False,
                "description": "Qiskit is an open-source SDK for working with quantum computers"
            }
        }

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a function",
    ["Home", "Code Generation", "Repository Training", "Sanskrit NLP", "Custom Model Training", "Data Comparison", "About"]
)

# Home page
if app_mode == "Home":
    st.title("AI Code Generation & Sanskrit NLP Platform")
    
    st.markdown("""
    ## Welcome to the Multi-Function AI Application
    
    This platform provides powerful AI tools for code generation and Sanskrit language processing:
    
    ### Key Features
    - **AI Code Generation**: Generate high-quality code based on your instructions
    - **Sanskrit Text Analysis**: Process and analyze Sanskrit texts with custom NLP capabilities
    - **Custom NLP Model Training**: Train specialized models for Sanskrit language processing
    - **Data Comparison**: Compare predicted weights vs actual data with statistical analysis
    - **Visualization**: Generate insights through comprehensive data visualizations
    
    ### Getting Started
    Select a function from the sidebar to begin exploring the capabilities of this platform.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("##### Code Generation\nGenerate code using powerful AI models")
        if st.button("Go to Code Generation", key="goto_code_gen"):
            app_mode = "Code Generation"
            st.rerun()
    
        st.info("##### Repository Training\nTrain on programming language source code from GitHub")
        if st.button("Go to Repository Training", key="goto_repo"):
            app_mode = "Repository Training"
            st.rerun()
    
    with col2:
        st.info("##### Sanskrit NLP\nProcess and analyze Sanskrit texts with custom linguistic tools")
        if st.button("Go to Sanskrit NLP", key="goto_nlp"):
            app_mode = "Sanskrit NLP"
            st.rerun()
            
        st.info("##### Model Training\nTrain custom NLP models for specialized Sanskrit tasks")
        if st.button("Go to Model Training", key="goto_training"):
            app_mode = "Custom Model Training"
            st.rerun()

# Sanskrit NLP page
elif app_mode == "Sanskrit NLP":
    st.title("Sanskrit NLP Processing")
    
    st.markdown("""
    This module provides specialized Natural Language Processing capabilities for Sanskrit text.
    You can input Sanskrit text directly or upload a file for analysis.
    """)
    
    # Input methods
    input_method = st.radio("Choose input method:", ["Enter Text", "Upload File"])
    
    if input_method == "Enter Text":
        sanskrit_input = st.text_area("Enter Sanskrit text:", height=200, value=st.session_state.sanskrit_text)
        if sanskrit_input:
            st.session_state.sanskrit_text = sanskrit_input
    else:
        uploaded_file = st.file_uploader("Upload a text file containing Sanskrit text:", type=["txt"])
        if uploaded_file is not None:
            # Read the file and convert to string
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            st.session_state.sanskrit_text = stringio.read()
            st.write("File uploaded successfully!")
            st.text_area("File contents:", st.session_state.sanskrit_text, height=200)
    
    # Analysis options
    if st.session_state.sanskrit_text:
        st.subheader("Analysis Options")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            tokenize = st.checkbox("Tokenization", value=True)
            grammar_analysis = st.checkbox("Grammar Analysis", value=True)
        
        with analysis_col2:
            pos_tagging = st.checkbox("POS Tagging", value=True)
            sentiment_analysis = st.checkbox("Semantic Analysis", value=False)
        
        if st.button("Process Text"):
            with st.spinner("Processing Sanskrit text..."):
                # Get analysis results
                results = {}
                
                if tokenize:
                    results["tokenization"] = sanskrit_nlp.tokenize(st.session_state.sanskrit_text)
                
                if grammar_analysis:
                    results["grammar"] = sanskrit_nlp.analyze_grammar(st.session_state.sanskrit_text)
                
                if pos_tagging:
                    results["pos_tagging"] = sanskrit_nlp.pos_tag(st.session_state.sanskrit_text)
                
                if sentiment_analysis:
                    results["semantics"] = sanskrit_nlp.analyze_semantics(st.session_state.sanskrit_text)
                
                st.session_state.nlp_results = results
        
        # Display results if available
        if st.session_state.nlp_results:
            st.subheader("Analysis Results")
            
            results_tabs = st.tabs(["Tokenization", "Grammar", "POS Tagging", "Semantics"])
            
            with results_tabs[0]:
                if "tokenization" in st.session_state.nlp_results:
                    tokens = st.session_state.nlp_results["tokenization"]
                    st.write(f"Found {len(tokens)} tokens:")
                    st.json(tokens)
                else:
                    st.info("Tokenization not performed")
            
            with results_tabs[1]:
                if "grammar" in st.session_state.nlp_results:
                    grammar_results = st.session_state.nlp_results["grammar"]
                    st.write("Grammar Analysis:")
                    st.json(grammar_results)
                else:
                    st.info("Grammar analysis not performed")
            
            with results_tabs[2]:
                if "pos_tagging" in st.session_state.nlp_results:
                    pos_results = st.session_state.nlp_results["pos_tagging"]
                    st.write("POS Tagging Results:")
                    
                    # Create a DataFrame for better display
                    pos_df = pd.DataFrame(pos_results, columns=["Token", "POS Tag"])
                    st.dataframe(pos_df)
                else:
                    st.info("POS tagging not performed")
            
            with results_tabs[3]:
                if "semantics" in st.session_state.nlp_results:
                    semantic_results = st.session_state.nlp_results["semantics"]
                    st.write("Semantic Analysis:")
                    st.json(semantic_results)
                else:
                    st.info("Semantic analysis not performed")
            
            # Export results
            st.subheader("Export Results")
            export_format = st.selectbox("Choose export format:", ["JSON", "CSV", "TXT"])
            
            if st.button("Export Results"):
                export_data = data_processor.format_export(st.session_state.nlp_results, export_format)
                st.download_button(
                    label=f"Download as {export_format}",
                    data=export_data,
                    file_name=f"sanskrit_analysis.{export_format.lower()}",
                    mime=data_processor.get_mime_type(export_format)
                )

# Custom Model Training page
elif app_mode == "Custom Model Training":
    st.title("Custom Sanskrit NLP Model Training")
    
    st.markdown("""
    This module allows you to train a custom NLP model specifically for Sanskrit language processing.
    Upload training data and configure model parameters to build specialized models.
    """)
    
    # Training data upload
    st.subheader("Training Data")
    
    data_format = st.radio("Data format:", ["CSV", "JSON", "Text Corpus"])
    
    uploaded_training = st.file_uploader(
        "Upload training data:", 
        type=["csv", "json", "txt"] if data_format == "Text Corpus" else 
              ["csv"] if data_format == "CSV" else ["json"]
    )
    
    if uploaded_training is not None:
        training_data = data_processor.load_data(uploaded_training, data_format)
        st.session_state.training_data = training_data
        
        if data_format == "CSV" or data_format == "JSON":
            st.write("Preview of training data:")
            st.dataframe(training_data.head() if isinstance(training_data, pd.DataFrame) else pd.DataFrame(training_data[:5]))
        else:
            st.write("Training corpus loaded successfully!")
            st.text_area("Preview:", training_data[:500] + "..." if len(training_data) > 500 else training_data, height=200)
    
    # Model configuration
    if st.session_state.training_data is not None:
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model type:", 
                ["Tokenizer", "POS Tagger", "Grammar Analyzer", "Semantic Analyzer"]
            )
            
            epochs = st.slider("Training epochs:", min_value=1, max_value=50, value=10)
        
        with col2:
            validation_split = st.slider("Validation split:", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            
            use_pretrained = st.checkbox("Use pretrained embeddings", value=True)
        
        # Advanced options
        with st.expander("Advanced Options"):
            batch_size = st.slider("Batch size:", min_value=8, max_value=128, value=32, step=8)
            learning_rate = st.number_input("Learning rate:", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
            dropout_rate = st.slider("Dropout rate:", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
        
        # Training
        if st.button("Train Model"):
            with st.spinner(f"Training {model_type} model... This may take a while"):
                # Configure training parameters
                params = {
                    "epochs": epochs,
                    "validation_split": validation_split,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "dropout_rate": dropout_rate,
                    "use_pretrained": use_pretrained
                }
                
                # Train the model
                model, training_history = model_trainer.train_model(
                    model_type,
                    st.session_state.training_data,
                    params
                )
                
                st.session_state.trained_model = {
                    "model": model,
                    "type": model_type,
                    "history": training_history,
                    "params": params
                }
                
                st.success(f"{model_type} model trained successfully!")
        
        # Display training results if available
        if st.session_state.trained_model:
            st.subheader("Training Results")
            
            # Plot training history
            history_fig = visualizer.plot_training_history(st.session_state.trained_model["history"])
            st.plotly_chart(history_fig)
            
            # Model evaluation
            st.write("Model evaluation:")
            eval_metrics = model_trainer.evaluate_model(
                st.session_state.trained_model["model"], 
                st.session_state.training_data, 
                validation_split
            )
            
            metrics_df = pd.DataFrame(list(eval_metrics.items()), columns=["Metric", "Value"])
            st.table(metrics_df)
            
            # Model export
            st.subheader("Export Model")
            if st.button("Export Trained Model"):
                model_binary = model_trainer.export_model(st.session_state.trained_model["model"])
                
                st.download_button(
                    label="Download Model",
                    data=model_binary,
                    file_name=f"sanskrit_{st.session_state.trained_model['type'].lower()}_model.pkl",
                    mime="application/octet-stream"
                )

# Code Generation page
elif app_mode == "Code Generation":
    st.title("AI-Powered Code Generation")
    
    st.markdown("""
    ## Intelligent Code Generation
    
    This module allows you to generate high-quality code using powerful AI models like OpenAI's GPT or Anthropic's Claude.
    You can generate code from scratch, explain existing code, improve code, or convert between programming languages.
    
    **Note**: This feature requires API keys to function. Please add your API keys in the settings below.
    """)
    
    # Check if API keys are available
    api_status = code_generator.check_api_keys()
    
    if not api_status["any_available"]:
        st.warning("⚠️ No API keys detected. Please add your OpenAI or Anthropic API key to use this feature.")
        
        # Create tabs for API key settings
        api_tabs = st.tabs(["OpenAI API", "Anthropic API"])
        
        with api_tabs[0]:
            st.write("#### OpenAI API Setup")
            st.markdown("""
            1. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)
            2. Copy your API key
            3. Add it as an environment variable named `OPENAI_API_KEY`
            """)
            
            # Input for OpenAI API key (for demonstration purposes - in production use secrets)
            openai_key = st.text_input("OpenAI API Key:", type="password", key="openai_key_input")
            if openai_key and st.button("Save OpenAI API Key", key="save_openai_key"):
                os.environ["OPENAI_API_KEY"] = openai_key
                st.success("✅ OpenAI API key saved successfully! Refreshing...")
                # Reinitialize the code generator with the new API key
                code_generator = CodeGenerator()
                st.rerun()
                
        with api_tabs[1]:
            st.write("#### Anthropic API Setup")
            st.markdown("""
            1. Get an API key from [Anthropic Console](https://console.anthropic.com/keys)
            2. Copy your API key
            3. Add it as an environment variable named `ANTHROPIC_API_KEY`
            """)
            
            # Input for Anthropic API key (for demonstration purposes - in production use secrets)
            anthropic_key = st.text_input("Anthropic API Key:", type="password", key="anthropic_key_input")
            if anthropic_key and st.button("Save Anthropic API Key", key="save_anthropic_key"):
                os.environ["ANTHROPIC_API_KEY"] = anthropic_key
                st.success("✅ Anthropic API key saved successfully! Refreshing...")
                # Reinitialize the code generator with the new API key
                code_generator = CodeGenerator()
                st.rerun()
    
    # Only show the code generation interface if at least one API is available
    if api_status["any_available"]:
        st.success(f"✅ API keys available: {', '.join([k for k, v in api_status.items() if v and k != 'any_available'])}")
        
        # Task selection
        code_task = st.selectbox(
            "Select task:",
            [
                "Generate New Code", 
                "Explain Code", 
                "Improve Code", 
                "Convert Code to Another Language",
                "Generate Unit Tests"
            ]
        )
        
        # Model selection
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            available_providers = []
            if api_status["openai_available"]:
                available_providers.append("openai")
            if api_status["anthropic_available"]:
                available_providers.append("anthropic")
                
            provider = st.selectbox(
                "Model provider:",
                available_providers,
                index=0 if "openai" in available_providers else 0
            )
        
        with model_col2:
            temperature = st.slider(
                "Temperature (creativity):",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Lower values produce more focused and deterministic outputs, higher values produce more creative outputs."
            )
        
        # Interface for selected task
        if code_task == "Generate New Code":
            st.subheader("Generate New Code")
            
            # Get the programming language
            lang_options = ["Python", "JavaScript", "TypeScript", "Java", "C#", "C++", "Go", "Rust", "PHP", "Ruby", "Swift", "Kotlin"]
            language = st.selectbox("Programming language:", lang_options)
            
            # Get the prompt and context
            prompt = st.text_area(
                "Describe what code you want to generate:",
                height=150,
                placeholder="E.g., Create a function to find the fibonacci sequence up to n terms"
            )
            
            # Optional context
            with st.expander("Add Context or Examples (Optional)"):
                context = st.text_area(
                    "Provide any additional context or example code:",
                    height=150,
                    placeholder="Provide examples or additional details to guide the code generation."
                )
            
            # Generate button
            if prompt and st.button("Generate Code"):
                with st.spinner("Generating code..."):
                    # Add language to prompt
                    full_prompt = f"Generate {language} code for the following: {prompt}"
                    
                    # Call the code generator
                    result = code_generator.generate_code(full_prompt, context, provider, None, temperature)
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        # Store the result in the session state
                        st.session_state.generated_code = result["generated_code"]
                        # Add to history
                        st.session_state.code_history.append({
                            "task": "Generate",
                            "language": language,
                            "prompt": prompt,
                            "result": result["generated_code"],
                            "model": result["model_used"]
                        })
            
        elif code_task == "Explain Code":
            st.subheader("Explain Code")
            
            # Get the code to explain
            code_to_explain = st.text_area(
                "Paste the code you want to explain:",
                height=250,
                placeholder="Paste your code here..."
            )
            
            # Explain button
            if code_to_explain and st.button("Explain Code"):
                with st.spinner("Analyzing code..."):
                    # Call the code generator to explain
                    result = code_generator.explain_code(code_to_explain, provider)
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        # Store the result in the session state
                        st.session_state.generated_code = result["generated_code"]
                        # Add to history
                        st.session_state.code_history.append({
                            "task": "Explain",
                            "code": code_to_explain,
                            "result": result["generated_code"],
                            "model": result["model_used"]
                        })
        
        elif code_task == "Improve Code":
            st.subheader("Improve Code")
            
            # Get the code to improve
            code_to_improve = st.text_area(
                "Paste the code you want to improve:",
                height=250,
                placeholder="Paste your code here..."
            )
            
            # Improvement focus areas
            improvement_areas = st.multiselect(
                "Focus on improving:",
                [
                    "Performance", 
                    "Readability", 
                    "Security", 
                    "Error Handling", 
                    "Documentation",
                    "Best Practices"
                ],
                default=["Performance", "Readability"]
            )
            
            # Improve button
            if code_to_improve and st.button("Improve Code"):
                with st.spinner("Improving code..."):
                    # Call the code generator to improve
                    result = code_generator.improve_code(
                        code_to_improve, 
                        improvement_areas,
                        provider
                    )
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        # Store the result in the session state
                        st.session_state.generated_code = result["generated_code"]
                        # Add to history
                        st.session_state.code_history.append({
                            "task": "Improve",
                            "code": code_to_improve,
                            "improvements": improvement_areas,
                            "result": result["generated_code"],
                            "model": result["model_used"]
                        })
        
        elif code_task == "Convert Code to Another Language":
            st.subheader("Convert Code to Another Language")
            
            # Get source and target languages
            lang_options = ["Python", "JavaScript", "TypeScript", "Java", "C#", "C++", "Go", "Rust", "PHP", "Ruby", "Swift", "Kotlin"]
            
            lang_col1, lang_col2 = st.columns(2)
            
            with lang_col1:
                source_lang = st.selectbox("Source language:", lang_options)
            
            with lang_col2:
                # Filter out the source language from options
                target_lang_options = [lang for lang in lang_options if lang != source_lang]
                target_lang = st.selectbox("Target language:", target_lang_options)
            
            # Get the code to convert
            code_to_convert = st.text_area(
                f"Paste the {source_lang} code you want to convert to {target_lang}:",
                height=250,
                placeholder=f"Paste your {source_lang} code here..."
            )
            
            # Convert button
            if code_to_convert and st.button("Convert Code"):
                with st.spinner(f"Converting {source_lang} to {target_lang}..."):
                    # Call the code generator to convert
                    result = code_generator.convert_code(
                        code_to_convert,
                        source_lang,
                        target_lang,
                        provider
                    )
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        # Store the result in the session state
                        st.session_state.generated_code = result["generated_code"]
                        # Add to history
                        st.session_state.code_history.append({
                            "task": "Convert",
                            "source_lang": source_lang,
                            "target_lang": target_lang,
                            "code": code_to_convert,
                            "result": result["generated_code"],
                            "model": result["model_used"]
                        })
        
        elif code_task == "Generate Unit Tests":
            st.subheader("Generate Unit Tests")
            
            # Get the programming language
            lang_options = ["Python", "JavaScript", "TypeScript", "Java", "C#", "C++", "Go", "Rust", "PHP", "Ruby", "Swift", "Kotlin"]
            language = st.selectbox("Programming language:", lang_options)
            
            # Get testing framework based on language
            testing_frameworks = {
                "Python": ["pytest", "unittest", "None (standard assertions)"],
                "JavaScript": ["Jest", "Mocha", "Jasmine", "None (standard assertions)"],
                "TypeScript": ["Jest", "Mocha", "Jasmine", "None (standard assertions)"],
                "Java": ["JUnit", "TestNG", "None (standard assertions)"],
                "C#": ["MSTest", "NUnit", "xUnit", "None (standard assertions)"],
                "Go": ["testing (standard library)", "testify", "None (standard assertions)"],
                "Rust": ["standard test module", "None (standard assertions)"],
                "PHP": ["PHPUnit", "Codeception", "None (standard assertions)"],
                "Ruby": ["RSpec", "Minitest", "None (standard assertions)"],
                "C++": ["Google Test", "Catch2", "Boost.Test", "None (standard assertions)"],
                "Swift": ["XCTest", "None (standard assertions)"],
                "Kotlin": ["JUnit", "Kotest", "None (standard assertions)"]
            }
            
            framework = st.selectbox(
                "Testing framework:",
                testing_frameworks.get(language, ["None (standard assertions)"])
            )
            
            # Get the code to test
            code_to_test = st.text_area(
                f"Paste the {language} code you want to generate tests for:",
                height=250,
                placeholder=f"Paste your {language} code here..."
            )
            
            # Generate button
            if code_to_test and st.button("Generate Tests"):
                with st.spinner(f"Generating {framework} tests for {language} code..."):
                    # Check if a framework was selected
                    test_framework = None if framework.startswith("None") else framework
                    
                    # Call the code generator to generate tests
                    result = code_generator.generate_unit_tests(
                        code_to_test,
                        language,
                        test_framework,
                        provider
                    )
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        # Store the result in the session state
                        st.session_state.generated_code = result["generated_code"]
                        # Add to history
                        st.session_state.code_history.append({
                            "task": "Tests",
                            "language": language,
                            "framework": framework,
                            "code": code_to_test,
                            "result": result["generated_code"],
                            "model": result["model_used"]
                        })
        
        # Display generated code if available
        if st.session_state.generated_code:
            st.subheader("Generated Output")
            st.code(st.session_state.generated_code)
            
            # Download button for the generated code
            download_name = f"generated_{code_task.lower().replace(' ', '_')}.txt"
            st.download_button(
                label="Download Generated Code",
                data=st.session_state.generated_code,
                file_name=download_name,
                mime="text/plain"
            )
        
        # Code history section
        if st.session_state.code_history:
            with st.expander("Code Generation History"):
                for i, history_item in enumerate(reversed(st.session_state.code_history)):
                    st.markdown(f"### {i+1}. {history_item['task']} ({history_item.get('model', 'AI')})")
                    
                    if history_item['task'] == "Generate":
                        st.markdown(f"**Prompt:** {history_item['prompt']}")
                        st.markdown(f"**Language:** {history_item['language']}")
                    elif history_item['task'] == "Convert":
                        st.markdown(f"**Conversion:** {history_item['source_lang']} → {history_item['target_lang']}")
                    elif history_item['task'] == "Improve":
                        st.markdown(f"**Improvements:** {', '.join(history_item['improvements'])}")
                    elif history_item['task'] == "Tests":
                        st.markdown(f"**Language:** {history_item['language']}")
                        st.markdown(f"**Framework:** {history_item['framework']}")
                    
                    with st.expander("View Output"):
                        st.code(history_item['result'])
                    
                    st.divider()

# Repository Training page
elif app_mode == "Repository Training":
    st.title("Code Repository Training")
    
    st.markdown("""
    ## Train on Programming Language Source Code
    
    This feature allows you to download, analyze, and learn from programming language source code repositories.
    You can use this to train code generation models without needing external API keys.
    
    ### How it works:
    1. Add GitHub repositories with source code
    2. Process and extract code snippets
    3. Analyze or search through the code
    4. Use the extracted code to generate similar code
    """)
    
    # Offline mode toggle
    offline_col1, offline_col2 = st.columns([1, 3])
    with offline_col1:
        offline_mode = st.checkbox("Offline Mode", value=st.session_state.repo_processor.offline_mode)
        if offline_mode != st.session_state.repo_processor.offline_mode:
            st.session_state.repo_processor.set_offline_mode(offline_mode)
            st.rerun()
    
    with offline_col2:
        if offline_mode:
            st.info("📥 Offline mode active: Using locally saved code data. Download features disabled.")
        else:
            st.info("🌐 Online mode active: Can download repositories from GitHub.")
    
    # Repository management section
    st.subheader("Repository Management")
    
    # Add a tab UI for online/offline options
    repo_tabs = st.tabs(["Add Repository", "Offline Data Management"])
    
    with repo_tabs[0]:
        # Add repository form - disabled in offline mode
        with st.form("add_repo_form"):
            st.write("Add a GitHub Repository")
            repo_url = st.text_input(
                "GitHub Repository URL:", 
                placeholder="https://github.com/username/repository",
                disabled=offline_mode
            )
            
            # Programming language selection
            lang_options = ["Python", "JavaScript", "TypeScript", "Java", "C#", "C++", "Go", "Rust", "PHP", "Ruby", "Swift", "Kotlin"]
            language = st.selectbox("Programming language:", lang_options)
            
            clone_depth = st.slider("Clone depth:", min_value=1, max_value=5, value=1, help="Shallow clone depth (1 is fastest)")
            
            submitted = st.form_submit_button("Add Repository", disabled=offline_mode)
            
            if offline_mode:
                st.info("⚠️ Repository download disabled in offline mode. Switch to online mode to add repositories.")
            
            if submitted and repo_url:
                with st.spinner(f"Cloning repository {repo_url}..."):
                    result = st.session_state.repo_processor.clone_repository(
                        repo_url=repo_url,
                        language=language,
                        depth=clone_depth
                    )
                    
                    if result["success"]:
                        st.success(result["message"])
                        # Store info in session state
                        repo_name = result["repo_name"]
                        if 'repos_data' not in st.session_state:
                            st.session_state.repos_data = {}
                        st.session_state.repos_data[repo_name] = {
                            "url": repo_url,
                            "language": language,
                            "processed": False
                        }
                    else:
                        st.error(result["message"])
    
    with repo_tabs[1]:
        st.write("#### Save/Load Code Data for Offline Use")
        
        offline_col1, offline_col2 = st.columns(2)
        
        with offline_col1:
            # Save data
            if st.button("Save All Data for Offline Use", disabled=len(st.session_state.repos_data) == 0):
                with st.spinner("Saving code data..."):
                    result = st.session_state.repo_processor.save_code_data()
                    if result["success"]:
                        st.success(result["message"])
                    else:
                        st.error(result["message"])
        
        with offline_col2:
            # Load data
            if st.button("Load Saved Data"):
                with st.spinner("Loading saved code data..."):
                    result = st.session_state.repo_processor.load_code_data()
                    if result["success"]:
                        st.success(result["message"])
                        # Update session state with loaded repositories
                        for repo_name in result.get("loaded_repos", []):
                            if repo_name in st.session_state.repo_processor.repos:
                                # Update or add to session state
                                repo_info = st.session_state.repo_processor.repos[repo_name]
                                if 'repos_data' not in st.session_state:
                                    st.session_state.repos_data = {}
                                st.session_state.repos_data[repo_name] = repo_info
                        
                        # Update code snippets
                        all_snippets = st.session_state.repo_processor.get_all_code_snippets()
                        st.session_state.code_snippets = all_snippets
                        
                        st.rerun()
                    else:
                        st.error(result["message"])
    
    # Process repositories section
    if st.session_state.repos_data:
        st.subheader("Process Repositories")
        
        # Show available repositories
        repos = list(st.session_state.repos_data.keys())
        selected_repo = st.selectbox("Select repository to process:", repos)
        
        if selected_repo:
            repo_info = st.session_state.repos_data[selected_repo]
            
            st.write(f"Repository: **{selected_repo}**")
            st.write(f"Language: **{repo_info['language']}**")
            st.write(f"URL: {repo_info['url']}")
            st.write(f"Processed: {'✅ Yes' if repo_info.get('processed', False) else '❌ No'}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Process button
                if st.button("Process Repository", key=f"process_{selected_repo}"):
                    with st.spinner(f"Processing {selected_repo}..."):
                        # Optional parameters for processing
                        max_files = 100  # Limit number of files to process
                        min_lines = 5    # Minimum lines per snippet
                        max_lines = 500  # Maximum lines per snippet
                        
                        result = st.session_state.repo_processor.process_repository(
                            repo_name=selected_repo,
                            max_files=max_files,
                            min_lines=min_lines,
                            max_lines=max_lines
                        )
                        
                        if result["success"]:
                            st.success(f"Processed {result['files_processed']} files, extracted {result['code_snippets']} code snippets")
                            # Update repo info
                            st.session_state.repos_data[selected_repo]["processed"] = True
                            st.session_state.repos_data[selected_repo]["files_processed"] = result["files_processed"]
                            st.session_state.repos_data[selected_repo]["code_snippets"] = result["code_snippets"]
                            # Get code snippets for this repo
                            snippets = st.session_state.repo_processor.get_all_code_snippets([repo_info["language"]])
                            st.session_state.code_snippets = snippets
                        else:
                            st.error(result["message"])
            
            with col2:
                # Remove button
                if st.button("Remove Repository", key=f"remove_{selected_repo}"):
                    with st.spinner(f"Removing {selected_repo}..."):
                        result = st.session_state.repo_processor.cleanup_repository(selected_repo)
                        
                        if result["success"]:
                            st.success(result["message"])
                            # Remove from session state
                            st.session_state.repos_data.pop(selected_repo, None)
                            st.rerun()
                        else:
                            st.error(result["message"])
        
        # Export code snippets section
        if st.session_state.code_snippets:
            st.subheader("Code Snippets")
            
            st.write(f"Total snippets: {len(st.session_state.code_snippets)}")
            
            # Export options
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if st.button("Export to JSON"):
                    with st.spinner("Exporting code snippets..."):
                        # Export to JSON file
                        result = st.session_state.repo_processor.export_to_json(
                            output_path="code_snippets.json",
                            languages=None,  # All languages
                            min_lines=5,
                            max_lines=None
                        )
                        
                        if result["success"]:
                            st.success(f"Exported {result['snippets_exported']} code snippets")
                            
                            # Create a download link for the JSON file
                            with open("code_snippets.json", "r") as f:
                                json_data = f.read()
                                
                            st.download_button(
                                label="Download JSON",
                                data=json_data,
                                file_name="code_snippets.json",
                                mime="application/json"
                            )
                        else:
                            st.error(result["message"])
            
            with export_col2:
                if st.button("Create DataFrame"):
                    with st.spinner("Creating DataFrame..."):
                        # Convert to DataFrame
                        df = st.session_state.repo_processor.export_to_dataframe()
                        
                        st.write("Preview of code snippets DataFrame:")
                        st.dataframe(df[["filename", "language", "line_count"]].head(10))
            
            # Code examples section
            with st.expander("View Sample Code Snippets"):
                num_samples = min(3, len(st.session_state.code_snippets))
                
                for i in range(num_samples):
                    snippet = st.session_state.code_snippets[i]
                    st.write(f"**{snippet['filename']}** ({snippet['language']}, {snippet['line_count']} lines)")
                    st.code(snippet['code'][:1000] + "..." if len(snippet['code']) > 1000 else snippet['code'], language=snippet['language'].lower())
                    st.divider()
        
        # Code search and similarity section
        if st.session_state.code_snippets:
            st.subheader("Code Search & Similarity")
            
            # Initialize vectorizer if not already done
            if st.button("Initialize Code Similarity Engine"):
                with st.spinner("Building code similarity model..."):
                    vectorizer = st.session_state.repo_processor.initialize_vectorizer()
                    if vectorizer:
                        st.success("Code similarity engine initialized successfully!")
                    else:
                        st.error("Failed to initialize similarity engine. Please process repositories first.")
            
            # Code similarity search
            st.write("#### Find Similar Code")
            query_code = st.text_area(
                "Enter code to find similar snippets:",
                height=150,
                placeholder="Paste code here to find similar examples..."
            )
            
            if query_code and st.button("Find Similar Code"):
                with st.spinner("Searching for similar code..."):
                    similar_codes = st.session_state.repo_processor.find_similar_code(query_code, top_n=5)
                    st.session_state.similar_code = similar_codes
                    
                    if similar_codes:
                        st.success(f"Found {len(similar_codes)} similar code snippets")
                    else:
                        st.info("No similar code snippets found")
            
            # Display similar code
            if st.session_state.similar_code:
                st.write("#### Similar Code Snippets")
                
                for i, result in enumerate(st.session_state.similar_code):
                    snippet = result["snippet"]
                    similarity = result["similarity_score"]
                    
                    expander_title = f"{i+1}. {snippet['filename']} (Similarity: {similarity:.2f})"
                    with st.expander(expander_title):
                        st.write(f"**Language:** {snippet['language']}")
                        st.write(f"**Path:** {snippet['path']}")
                        st.write(f"**Lines:** {snippet['line_count']}")
                        st.code(snippet['code'], language=snippet['language'].lower())

            # Code completion section
            st.write("#### Code Completion")
            st.write("Use existing code to generate completions")
            
            partial_code = st.text_area(
                "Enter partial code to complete:",
                height=150,
                placeholder="Start writing some code here..."
            )
            
            if partial_code:
                lang_options = ["Python", "JavaScript", "TypeScript", "Java", "C#", "C++"]
                completion_lang = st.selectbox("Completion language:", lang_options, key="completion_lang")
                
                if st.button("Generate Completion"):
                    with st.spinner("Generating code completion..."):
                        completions = st.session_state.repo_processor.generate_code_completion(
                            partial_code=partial_code,
                            language=completion_lang,
                            max_candidates=3,
                            min_similarity=0.2
                        )
                        
                        if completions:
                            st.success(f"Generated {len(completions)} completion candidates")
                            
                            for i, candidate in enumerate(completions):
                                score = candidate["score"]
                                code = candidate["code"]
                                
                                with st.expander(f"Completion #{i+1} (Score: {score:.2f})"):
                                    st.code(code, language=completion_lang.lower())
                        else:
                            st.info("Could not generate completions. Try adding more repositories or changing the partial code.")
    else:
        st.info("No repositories added yet. Add a repository using the form above.")

# Data Comparison page
elif app_mode == "Data Comparison":
    st.title("Data Comparison Analysis")
    
    st.markdown("""
    This module helps you compare and analyze different datasets, such as predicted weights versus actual values.
    Upload your data files and explore statistical comparisons and visualizations.
    """)
    
    # Data upload
    st.subheader("Upload Data for Comparison")
    
    upload_col1, upload_col2 = st.columns(2)
    
    with upload_col1:
        st.write("Dataset 1 (e.g., Predicted)")
        data1_format = st.selectbox("Format for Dataset 1:", ["CSV", "Excel", "JSON"], key="data1_format")
        data1_file = st.file_uploader(
            "Upload Dataset 1:", 
            type=["csv"] if data1_format == "CSV" else 
                  ["xlsx", "xls"] if data1_format == "Excel" else ["json"],
            key="data1_upload"
        )
    
    with upload_col2:
        st.write("Dataset 2 (e.g., Actual)")
        data2_format = st.selectbox("Format for Dataset 2:", ["CSV", "Excel", "JSON"], key="data2_format")
        data2_file = st.file_uploader(
            "Upload Dataset 2:", 
            type=["csv"] if data2_format == "CSV" else 
                  ["xlsx", "xls"] if data2_format == "Excel" else ["json"],
            key="data2_upload"
        )
    
    # Process uploaded data
    if data1_file and data2_file:
        data1 = data_processor.load_data(data1_file, data1_format)
        data2 = data_processor.load_data(data2_file, data2_format)
        
        st.session_state.comparison_data = {
            "data1": data1,
            "data2": data2,
            "data1_name": data1_file.name,
            "data2_name": data2_file.name
        }
        
        # Display data previews
        st.subheader("Data Previews")
        
        preview_tabs = st.tabs([f"Dataset 1: {data1_file.name}", f"Dataset 2: {data2_file.name}"])
        
        with preview_tabs[0]:
            st.dataframe(data1.head() if isinstance(data1, pd.DataFrame) else pd.DataFrame(data1[:5]))
            st.write(f"Shape: {data1.shape if hasattr(data1, 'shape') else (len(data1), len(data1[0]) if len(data1) > 0 else 0)}")
        
        with preview_tabs[1]:
            st.dataframe(data2.head() if isinstance(data2, pd.DataFrame) else pd.DataFrame(data2[:5]))
            st.write(f"Shape: {data2.shape if hasattr(data2, 'shape') else (len(data2), len(data2[0]) if len(data2) > 0 else 0)}")
        
        # Comparison configuration
        st.subheader("Comparison Configuration")
        
        # Only proceed if both datasets are DataFrames
        if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
            # Column mapping
            st.write("Map columns for comparison:")
            
            # Get column options from both DataFrames
            data1_cols = data1.columns.tolist()
            data2_cols = data2.columns.tolist()
            
            mapping_cols = st.columns(3)
            
            with mapping_cols[0]:
                key_col1 = st.selectbox("Key column in Dataset 1:", data1_cols)
                key_col2 = st.selectbox("Key column in Dataset 2:", data2_cols)
            
            with mapping_cols[1]:
                value_col1 = st.selectbox("Value column in Dataset 1 (e.g., predicted):", data1_cols)
                value_col2 = st.selectbox("Value column in Dataset 2 (e.g., actual):", data2_cols)
            
            with mapping_cols[2]:
                comparison_method = st.selectbox(
                    "Comparison method:",
                    ["Difference (A-B)", "Percentage Difference", "Ratio (A/B)", "Statistical Tests"]
                )
            
            # Run comparison
            if st.button("Run Comparison"):
                with st.spinner("Running comparison analysis..."):
                    comparison_results = data_processor.compare_datasets(
                        data1, data2,
                        key_col1, key_col2,
                        value_col1, value_col2,
                        comparison_method
                    )
                    
                    st.session_state.comparison_results = comparison_results
            
            # Display comparison results if available
            if st.session_state.comparison_results:
                st.subheader("Comparison Results")
                
                # Display results table
                results_df = st.session_state.comparison_results["results_df"]
                st.dataframe(results_df)
                
                # Summary statistics
                st.write("Summary Statistics:")
                summary_df = st.session_state.comparison_results["summary"]
                st.table(summary_df)
                
                # Visualizations
                st.subheader("Visualizations")
                
                viz_tabs = st.tabs(["Comparison Plot", "Distribution", "Correlation", "Error Analysis"])
                
                with viz_tabs[0]:
                    comparison_fig = visualizer.plot_comparison(
                        results_df, 
                        key_col1, 
                        f"{value_col1} ({data1_file.name})", 
                        f"{value_col2} ({data2_file.name})"
                    )
                    st.plotly_chart(comparison_fig, use_container_width=True)
                
                with viz_tabs[1]:
                    dist_fig = visualizer.plot_distributions(
                        results_df,
                        f"{value_col1} ({data1_file.name})",
                        f"{value_col2} ({data2_file.name})"
                    )
                    st.plotly_chart(dist_fig, use_container_width=True)
                
                with viz_tabs[2]:
                    corr_fig = visualizer.plot_correlation(
                        results_df,
                        f"{value_col1} ({data1_file.name})",
                        f"{value_col2} ({data2_file.name})"
                    )
                    st.plotly_chart(corr_fig, use_container_width=True)
                
                with viz_tabs[3]:
                    error_fig = visualizer.plot_error_analysis(
                        results_df,
                        key_col1,
                        "difference" if "difference" in results_df.columns else "percent_diff" if "percent_diff" in results_df.columns else "ratio"
                    )
                    st.plotly_chart(error_fig, use_container_width=True)
                
                # Export results
                st.subheader("Export Results")
                
                export_format = st.selectbox("Export format:", ["CSV", "Excel", "JSON"])
                
                if st.button("Export Results"):
                    export_data = data_processor.export_comparison_results(
                        st.session_state.comparison_results["results_df"],
                        export_format
                    )
                    
                    st.download_button(
                        label=f"Download as {export_format}",
                        data=export_data,
                        file_name=f"comparison_results.{export_format.lower()}",
                        mime=data_processor.get_mime_type(export_format)
                    )
        else:
            st.error("Both datasets must be in tabular format (CSV, Excel) for comparison")

# About page
elif app_mode == "About":
    st.title("About This Application")
    
    st.markdown("""
    ## AI Code Generation & Sanskrit NLP Platform
    
    This application provides comprehensive tools for code generation, repository analysis, 
    and Sanskrit language processing with both online and offline capabilities.
    
    ### Key Features
    
    #### Code Generation
    - Generate high-quality code based on natural language prompts
    - Leverage advanced AI models like GPT-4 or Claude (when API keys are provided)
    - Use local repository training for offline code generation
    - Explain and optimize existing code
    - Convert code between different programming languages
    
    #### Repository Training
    - Download and analyze GitHub repositories
    - Extract code snippets for offline use
    - Find similar code based on patterns
    - Generate code completions without external APIs
    - Works completely offline with saved repository data
    
    #### Sanskrit NLP Module
    - Custom tokenization for Sanskrit text
    - Grammar rule implementation and analysis
    - Parts-of-speech tagging
    - Semantic analysis
    
    #### Custom Model Training
    - Build specialized NLP models for Sanskrit
    - Train models on custom datasets
    - Configure advanced training parameters
    - Export trained models for later use
    
    #### Data Comparison
    - Compare predicted vs actual data
    - Perform statistical analysis on datasets
    - Visualize comparisons with interactive charts
    - Export comparison results
    
    ### Operating Modes
    
    This application offers multiple operation modes:
    
    #### Online Mode
    - Can clone and analyze GitHub repositories
    - Optionally uses OpenAI or Anthropic APIs (if API keys are provided)
    - Full functionality for all features
    
    #### Offline Mode
    - Works without internet connectivity
    - Uses locally saved code repositories
    - No dependency on external cloud services
    - All code analysis and generation happens locally
    
    ### Technical Implementation
    
    This application is built with:
    - Streamlit for the web interface
    - GitPython for repository management
    - NLTK and custom algorithms for NLP
    - Optional integrations with OpenAI and Anthropic
    - Pandas and NumPy for data manipulation
    - Scikit-learn for code similarity analysis
    - Plotly for interactive visualizations
    
    ### Default Repositories
    
    The application comes with:
    - **Qiskit**: IBM's open-source quantum computing framework
    - Add more repositories through the Repository Training section
    
    ### Usage Guidelines
    
    For best results:
    - Save repository data for offline use
    - Use UTF-8 encoded text files for Sanskrit input
    - Ensure consistent formats across comparison datasets
    - Provide sufficient training data for model development
    """)

    st.info("Created with ❤️ for AI code generation, repository training, and Sanskrit language processing")
