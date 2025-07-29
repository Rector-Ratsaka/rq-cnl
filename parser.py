import pandas as pd
import re
from collections import defaultdict

def parse_mapping_string(mapping_str):
    """
    Parse a mapping string and extract EC and PC components
    
    Args:
        mapping_str: String containing mappings like "[text](EC1) ; [text](PC1)"
    
    Returns:
        dict: Dictionary with EC and PC keys and their corresponding text values
    """
    # Pattern to match [text](CODE) format
    pattern = r'\[([^\]]+)\]\(([EP]C\d+)\)'
    matches = re.findall(pattern, mapping_str)
    
    result = {}
    for text, code in matches:
        result[code] = text
    
    return result

def expand_mapping_columns(data, max_ec=5, max_pc=2):
    """
    Expand mapping column into separate EC and PC columns
    
    Args:
        data: List of dictionaries or DataFrame with research_question, templated_question, mapping columns
        max_ec: Maximum number of EC columns to create (default: 5)
        max_pc: Maximum number of PC columns to create (default: 2)
    
    Returns:
        DataFrame with expanded columns
    """
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Parse all mappings
    parsed_mappings = []
    for mapping in df['mapping']:
        parsed = parse_mapping_string(mapping)
        parsed_mappings.append(parsed)
    
    # Create fixed column structure
    ec_codes = [f'EC{i}' for i in range(1, max_ec + 1)]
    pc_codes = [f'PC{i}' for i in range(1, max_pc + 1)]
    all_codes = ec_codes + pc_codes
    
    # Create new DataFrame with expanded columns
    result_df = df[['research_question', 'templated_question']].copy()
    
    # Add columns for each code (fixed structure)
    for code in all_codes:
        result_df[code] = [parsed.get(code, '') for parsed in parsed_mappings]
    
    return result_df


# Main execution - load from CSV file
def main():
    # Specify your input CSV file path
    input_file = 'cnl_output/llama_templates_mappings.csv'  # Change this to your actual file path
    output_file = 'cnl_output/llama_templates_exp_mappings.csv'  # Output file name
    
    try:
        # Load and process the CSV file
        print(f"Loading data from {input_file}...")
        expanded_df = process_csv_file(input_file)
       
        
        # Save to CSV
        expanded_df.to_csv(output_file, index=False)
        print(f"\nExpanded data saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found. Please check the file path.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

# Function to load from CSV file
def process_csv_file(filename):
    """
    Process a CSV file with mapping data
    
    Args:
        filename: Path to CSV file with columns: research_question, templated_question, mapping
    
    Returns:
        DataFrame with expanded columns
    """
    df = pd.read_csv(filename)
    return expand_mapping_columns(df)

if __name__ == "__main__":
    main()


