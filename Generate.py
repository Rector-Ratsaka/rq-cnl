import torch
import pandas as pd
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ChunkingLib import extract_EC_chunks, extract_PC_chunks

@dataclass
class QuestionGenerationConfig:
	"""Configuration for template generation from research questions"""
	grammar_correction: bool = True  # Set to False to skip correction step

class CNLTemplateGenerator:
	def __init__(self, config: QuestionGenerationConfig):
		self.config = config
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f"Using device: {self.device}")

		if config.grammar_correction:
			print("Loading grammar correction model...")
			self.tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
			self.model = AutoModelForSeq2SeqLM.from_pretrained("grammarly/coedit-large").to(self.device)
			print("Grammar correction model loaded.")

	def correct_grammar(self, sentence: str) -> str:
		input_text = f'Fix grammatical errors in this sentence: {sentence}'
		input_ids = self.tokenizer(input_text, return_tensors="pt", truncation=True).input_ids.to(self.device)
		outputs = self.model.generate(input_ids, max_length=256)
		corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
		return corrected_text

	def extract_template(self, question: str) -> str:
		return extract_PC_chunks(extract_EC_chunks(question))

def create_output_directory():
	output_dir = Path.cwd() / "output"
	output_dir.mkdir(exist_ok=True)
	return output_dir

def process_csv_file(file_path: Path, config: QuestionGenerationConfig):
	print(f"\nProcessing file: {file_path}")
	df = pd.read_csv(file_path)

	if 'research_question' not in df.columns:
		print(f"Missing 'research_question' column in {file_path.name}")
		return

	df = df[['research_question']].dropna().drop_duplicates()
	df = df.rename(columns={'research_question': 'InputTexts'})

	print(f"Found {len(df)} unique research questions.")

	generator = CNLTemplateGenerator(config)

	# Step 1: Grammar correction
	if config.grammar_correction:
		print("Correcting grammar...")
		df['GenQuestion'] = df['InputTexts'].apply(generator.correct_grammar)
	else:
		df['GenQuestion'] = df['InputTexts']

	# Step 2: Extract CNL templates
	print("Extracting templates...")
	df['TemplatedQuestion'] = df['GenQuestion'].apply(generator.extract_template)

	# Step 3: Save output
	output_dir = create_output_directory()
	output_path = output_dir / f"{file_path.stem}-CNL_Templates.csv"
	df.to_csv(output_path, index=False, encoding='utf-8')
	print(f"Saved CNL templates to: {output_path}")

def main():
	input_dir = Path.cwd() / "rqs_50"
	csv_files = list(input_dir.glob("*.csv"))
	if not csv_files:
		print(f"No CSV files found in {input_dir}")
		return

	config = QuestionGenerationConfig(grammar_correction=True)  # Toggle grammar correction here
	for file in csv_files:
		process_csv_file(file, config)

if __name__ == '__main__':
	main()