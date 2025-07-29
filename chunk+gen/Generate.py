import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from ChunkingLib import extract_EC_chunks, extract_PC_chunks

@dataclass
class CNLTemplateGenerator:
	def __init__(self):
		print("Initialized CNL Template Generator.")

	def extract_template(self, question: str) -> str:
		return extract_PC_chunks(extract_EC_chunks(question))

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

	# Extract CNL templates 
	print("Extracting templates...")
	df['templated_question'] = df['research_question'].apply(generator.extract_template)

	# Sort by length of the templated question
	df['template_length'] = df['templated_question'].str.len()
	df = df.sort_values(by='template_length', ascending=True).drop(columns='template_length')

	# Save output
	output_dir = create_output_directory()
	output_path = output_dir / "mistral_mod.csv"
	df.to_csv(output_path, index=False, encoding='utf-8')
	print(f"Saved CNL templates to: {output_path}")

def main():
	file_path = Path("research_questions/mistral_rqs.csv")
	if not file_path.exists():
		print(f"File not found: {file_path}")
		return

	process_csv_file(file_path)

if __name__ == '__main__':
	main()