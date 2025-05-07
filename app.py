import streamlit as st
import pandas as pd
import numpy as np
import io
from utils.sanskrit_nlp import SanskritNLP
from utils.data_processor import DataProcessor
from utils.visualizer import Visualizer
from utils.model_trainer import ModelTrainer

# Set page configuration
st.set_page_config(
    page_title="Sanskrit NLP & Data Analysis Tool",
    page_icon="🕉️",
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

# Create instances of utility classes
sanskrit_nlp = SanskritNLP()
data_processor = DataProcessor()
visualizer = Visualizer()
model_trainer = ModelTrainer()

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a function",
    ["Home", "Sanskrit NLP", "Custom Model Training", "Data Comparison", "About"]
)

# Home page
if app_mode == "Home":
    st.title("Sanskrit NLP & Data Analysis Platform")
    
    st.markdown("""
    ## Welcome to the Multi-Function AI Application
    
    This platform provides powerful tools for Sanskrit language processing and data analysis:
    
    ### Key Features
    - **Sanskrit Text Analysis**: Process and analyze Sanskrit texts with custom NLP capabilities
    - **Custom NLP Model Training**: Train specialized models for Sanskrit language processing
    - **Data Comparison**: Compare predicted weights vs actual data with statistical analysis
    - **Visualization**: Generate insights through comprehensive data visualizations
    
    ### Getting Started
    Select a function from the sidebar to begin exploring the capabilities of this platform.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("##### Sanskrit NLP\nProcess and analyze Sanskrit texts with custom linguistic tools")
        if st.button("Go to Sanskrit NLP", key="goto_nlp"):
            app_mode = "Sanskrit NLP"
            st.rerun()
            
    with col2:
        st.info("##### Model Training\nTrain custom NLP models for specialized Sanskrit tasks")
        if st.button("Go to Model Training", key="goto_training"):
            app_mode = "Custom Model Training"
            st.rerun()
            
    with col3:
        st.info("##### Data Comparison\nCompare and analyze various data sources with statistical tools")
        if st.button("Go to Data Comparison", key="goto_comparison"):
            app_mode = "Data Comparison"
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
    ## Sanskrit NLP & Data Analysis Platform
    
    This application combines specialized Sanskrit language processing with powerful data analysis capabilities.
    
    ### Key Features
    
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
    
    ### Technical Implementation
    
    This application is built with:
    - Streamlit for the web interface
    - NLTK and spaCy for NLP foundations
    - Custom Sanskrit processing extensions
    - Pandas and NumPy for data manipulation
    - Matplotlib and Plotly for visualizations
    - scikit-learn for machine learning components
    
    ### Usage Guidelines
    
    For best results:
    - Use UTF-8 encoded text files for Sanskrit input
    - Ensure consistent formats across comparison datasets
    - Provide sufficient training data for model development
    """)

    st.info("Created with ❤️ for Sanskrit language processing and data analysis")
