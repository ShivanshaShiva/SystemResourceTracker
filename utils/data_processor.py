import pandas as pd
import numpy as np
import io
import json
from scipy import stats
import pickle
import zipfile
import base64

class DataProcessor:
    """
    Class for handling data processing tasks such as loading, comparing,
    and exporting data in various formats.
    """
    
    def __init__(self):
        """Initialize the DataProcessor class."""
        self.supported_formats = ["CSV", "Excel", "JSON", "TXT"]
    
    def load_data(self, file, format_type):
        """
        Load data from various file formats.
        
        Args:
            file: File object to load
            format_type (str): Format of the file (CSV, Excel, JSON, TXT)
            
        Returns:
            Various: Loaded data (DataFrame for tabular data, dict for JSON, str for text)
        """
        if format_type == "CSV":
            return pd.read_csv(file)
        
        elif format_type == "Excel":
            return pd.read_excel(file)
        
        elif format_type == "JSON":
            return pd.read_json(file)
        
        elif format_type == "Text Corpus" or format_type == "TXT":
            return file.getvalue().decode("utf-8")
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def compare_datasets(self, data1, data2, key_col1, key_col2, value_col1, value_col2, comparison_method):
        """
        Compare two datasets based on specified columns.
        
        Args:
            data1 (DataFrame): First dataset
            data2 (DataFrame): Second dataset
            key_col1 (str): Key column name in the first dataset
            key_col2 (str): Key column name in the second dataset
            value_col1 (str): Value column name in the first dataset
            value_col2 (str): Value column name in the second dataset
            comparison_method (str): Method of comparison
            
        Returns:
            dict: Comparison results including a DataFrame and summary statistics
        """
        # Rename columns to avoid conflicts
        data1_renamed = data1.rename(columns={key_col1: "key", value_col1: "value1"})
        data2_renamed = data2.rename(columns={key_col2: "key", value_col2: "value2"})
        
        # Merge datasets on the key column
        merged = pd.merge(data1_renamed[["key", "value1"]], 
                          data2_renamed[["key", "value2"]], 
                          on="key", how="inner")
        
        # Calculate comparison metric based on selected method
        if comparison_method == "Difference (A-B)":
            merged["difference"] = merged["value1"] - merged["value2"]
            merged["abs_difference"] = abs(merged["difference"])
            
        elif comparison_method == "Percentage Difference":
            # Handle division by zero
            merged["percent_diff"] = np.where(
                merged["value2"] != 0,
                ((merged["value1"] - merged["value2"]) / merged["value2"]) * 100,
                np.nan
            )
            merged["abs_percent_diff"] = abs(merged["percent_diff"])
            
        elif comparison_method == "Ratio (A/B)":
            # Handle division by zero
            merged["ratio"] = np.where(
                merged["value2"] != 0,
                merged["value1"] / merged["value2"],
                np.nan
            )
        
        elif comparison_method == "Statistical Tests":
            # No additional column needed, statistical tests will be performed separately
            pass
        
        # Calculate summary statistics
        summary = {}
        
        # Common statistics for all methods
        summary["count"] = len(merged)
        summary["mean_value1"] = merged["value1"].mean()
        summary["mean_value2"] = merged["value2"].mean()
        
        # Method-specific statistics
        if comparison_method == "Difference (A-B)":
            summary["mean_diff"] = merged["difference"].mean()
            summary["median_diff"] = merged["difference"].median()
            summary["std_diff"] = merged["difference"].std()
            summary["max_abs_diff"] = merged["abs_difference"].max()
            
        elif comparison_method == "Percentage Difference":
            summary["mean_percent_diff"] = merged["percent_diff"].mean()
            summary["median_percent_diff"] = merged["percent_diff"].median()
            summary["std_percent_diff"] = merged["percent_diff"].std()
            summary["max_abs_percent_diff"] = merged["abs_percent_diff"].max()
            
        elif comparison_method == "Ratio (A/B)":
            summary["mean_ratio"] = merged["ratio"].mean()
            summary["median_ratio"] = merged["ratio"].median()
            summary["std_ratio"] = merged["ratio"].std()
        
        # Statistical tests
        if comparison_method == "Statistical Tests" or True:  # Include tests for all methods
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(merged["value1"], merged["value2"])
            summary["t_statistic"] = t_stat
            summary["p_value"] = p_value
            summary["significant_diff"] = p_value < 0.05
            
            # Correlation
            correlation = merged["value1"].corr(merged["value2"])
            summary["correlation"] = correlation
        
        # Create summary DataFrame
        summary_df = pd.DataFrame({
            "Metric": list(summary.keys()),
            "Value": list(summary.values())
        })
        
        # Rename the key column back to its original name for clarity
        results_df = merged.rename(columns={"key": key_col1})
        
        return {
            "results_df": results_df,
            "summary": summary_df
        }
    
    def format_export(self, data, format_type):
        """
        Format data for export in various formats.
        
        Args:
            data: Data to format
            format_type (str): Format to export as
            
        Returns:
            Various: Formatted data ready for export
        """
        if format_type == "JSON":
            # Convert to JSON
            if isinstance(data, dict):
                return json.dumps(data, indent=4)
            elif isinstance(data, pd.DataFrame):
                return data.to_json(orient="records", indent=4)
            else:
                return json.dumps(str(data))
        
        elif format_type == "CSV":
            # Convert to CSV
            if isinstance(data, pd.DataFrame):
                return data.to_csv(index=False)
            elif isinstance(data, dict):
                return pd.DataFrame(data).to_csv(index=False)
            else:
                # Convert to string and output as CSV
                return "data\n" + str(data)
        
        elif format_type == "TXT":
            # Convert to plain text
            if isinstance(data, dict):
                return "\n".join([f"{k}: {v}" for k, v in data.items()])
            elif isinstance(data, pd.DataFrame):
                return data.to_string()
            else:
                return str(data)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def export_comparison_results(self, results_df, format_type):
        """
        Export comparison results in various formats.
        
        Args:
            results_df (DataFrame): Comparison results DataFrame
            format_type (str): Format to export as
            
        Returns:
            Various: Formatted data ready for export
        """
        if format_type == "CSV":
            buffer = io.StringIO()
            results_df.to_csv(buffer, index=False)
            return buffer.getvalue()
        
        elif format_type == "Excel":
            buffer = io.BytesIO()
            results_df.to_excel(buffer, index=False)
            return buffer.getvalue()
        
        elif format_type == "JSON":
            return results_df.to_json(orient="records", indent=4)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def get_mime_type(self, format_type):
        """
        Get the MIME type for a given file format.
        
        Args:
            format_type (str): Format type
            
        Returns:
            str: MIME type
        """
        mime_types = {
            "CSV": "text/csv",
            "Excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "JSON": "application/json",
            "TXT": "text/plain"
        }
        
        return mime_types.get(format_type, "application/octet-stream")
