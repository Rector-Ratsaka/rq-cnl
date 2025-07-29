import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from ChunkingLib import extract_EC_chunks, extract_PC_chunks, extract_EC_chunks_with_mapping, extract_PC_chunks_with_mapping

@dataclass
class CNLTemplateGenerator:
    def __init__(self):
        print("Initialized CNL Template Generator.")

    def extract_template(self, question: str) -> str:
        """Original method for backward compatibility"""
        return extract_PC_chunks(extract_EC_chunks(question))

    def extract_template_with_mapping(self, question: str) -> tuple:
        """
        Extract template and create mapping between original text and EC/PC markers
        Returns: (templated_question, mapping_string)
        """
        original_question = question
        mappings = {}
        
        # First pass: Extract EC chunks with mapping
        question_with_ec, ec_mappings = extract_EC_chunks_with_mapping(question, mappings)
        
        # Second pass: Extract PC chunks with mapping
        final_templated, pc_mappings = extract_PC_chunks_with_mapping(question_with_ec, mappings)
        
        # Create mapping string
        mapping_string = self._create_mapping_string(mappings)
        
        return final_templated, mapping_string

    def _create_mapping_string(self, mappings: dict) -> str:
        """
        Create the mapping string in the format requested
        Format: [text](EC1) ; [text](PC1) ; [text](EC2) etc.
        """
        if not mappings:
            return ""
            
        mapping_parts = []
        
        # Sort mappings by marker type and number (EC1, EC2, PC1, PC2, etc.)
        def sort_key(item):
            marker = item[0]
            # Extract type (EC/PC) and number
            marker_type = marker[:2]  # EC or PC
            marker_num = int(marker[2:])  # number
            # Sort EC first, then PC, then by number
            type_priority = 0 if marker_type == 'EC' else 1
            return (type_priority, marker_num)
        
        sorted_mappings = sorted(mappings.items(), key=sort_key)
        
        for marker, text in sorted_mappings:
            mapping_parts.append(f"[{text}]({marker})")
        
        return " ; ".join(mapping_parts)

def create_output_directory():
    output_dir = Path.cwd() / "cnl_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir

def process_csv_file(file_path: Path):
    print(f"\nProcessing file: {file_path}")
    df = pd.read_csv(file_path)

    if 'research_question' not in df.columns:
        print(f"Missing 'research_question' column in {file_path.name}")
        return

    df = df[['research_question']].dropna().drop_duplicates()

    print(f"Found {len(df)} unique research questions.")

    generator = CNLTemplateGenerator()

    # Extract CNL templates with mappings
    print("Extracting templates and mappings...")
    
    template_mapping_results = df['research_question'].apply(
        lambda x: generator.extract_template_with_mapping(x)
    )
    
    df['templated_question'] = template_mapping_results.apply(lambda x: x[0])
    df['mapping'] = template_mapping_results.apply(lambda x: x[1])

    # Sort by length of the templated question
    df['template_length'] = df['templated_question'].str.len()
    df = df.sort_values(by='template_length', ascending=True).drop(columns='template_length')

    # Reorder columns to match requested format
    df = df[['research_question', 'templated_question', 'mapping']]

    # Save output
    output_dir = create_output_directory()
    output_path = output_dir / "llama_templates_mappings.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved CNL templates with mappings to: {output_path}")
    

def main():
    file_path = Path("research_questions/llama_rqs.csv")
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    process_csv_file(file_path)

if __name__ == '__main__':
    main()