import os
import csv
import time
import openai
from pathlib import Path

# You'll need to install the OpenAI Python library
# pip install openai

# Set your OpenAI API key
# You can either set it directly here, or set it as an environment variable OPENAI_API_KEY
openai.api_key = ""  
# Or use: openai.api_key = os.environ["OPENAI_API_KEY"]

def process_spanish_words(input_file):
    """
    Process a list of Spanish words and enhance with translations and examples
    using OpenAI's GPT-4.
    
    Parameters:
    - input_file: Path to a text file with one Spanish word per line
    """
    # Set output file in the same directory as the input file
    input_path = Path(input_file)
    output_file = str(input_path.with_name(f"{input_path.stem}_enhanced.csv"))
    
    # Try to get API key
    api_key = openai.api_key#os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ").strip()
        if not api_key:
            print("No API key provided. Exiting.")
            return False
    
    openai.api_key = api_key
    
    # Read Spanish words from the file
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            spanish_words = [line.strip() for line in file if line.strip()]
    except Exception as e:
        print(f"Error reading input file: {e}")
        return False
    
    print(f"Found {len(spanish_words)} Spanish words to process.")
    
    # Prepare the CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Spanish', 'English', 'SpanishExample', 'EnglishExample']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process words in batches to minimize API calls
        batch_size = 100
        for i in range(0, len(spanish_words), batch_size):
            batch = spanish_words[i:i+batch_size]
            print(f"Processing words {i+1}-{i+len(batch)} of {len(spanish_words)}...")
            
            # Create the prompt for the API
            prompt = "For each of the following Spanish words, provide:\n"
            prompt += "1. An English translation\n"
            prompt += "2. A simple example sentence in Spanish\n"
            prompt += "3. An English translation of the example sentence\n\n"
            prompt += "Format the output as a CSV with columns: Spanish,English,SpanishExample,EnglishExample\n"
            prompt += "Words to translate:\n"
            for word in batch:
                prompt += f"- {word}\n"
            
            try:
                # Call the OpenAI API
                response = openai.chat.completions.create(
                    model="o4-mini",  # You can also use "gpt-3.5-turbo" for a lower-cost option
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides accurate translations and examples for Spanish words."},
                        {"role": "user", "content": prompt}
                    ],
                    #temperature=0.3  # Lower temperature for more consistent results
                )
                
                # Parse the response
                content = response.choices[0].message.content.strip()
                print(content)
                # Skip the header if present
                if content.startswith("Spanish,English"):
                    content = content.split('\n', 1)[1]
                
                # Process each line
                for line in content.split('\n'):
                    if line.strip():
                        try:
                            # Handle potential quoting issues in the CSV
                            parts = line.split(',')
                            if len(parts) >= 4:
                                writer.writerow({
                                    'Spanish': parts[0].strip(),
                                    'English': parts[1].strip(),
                                    'SpanishExample': parts[2].strip(),
                                    'EnglishExample': ','.join(parts[3:]).strip()  # Join remaining parts in case the English example contains commas
                                })
                            else:
                                print(f"Warning: Could not parse line: {line}")
                        except Exception as e:
                            print(f"Error processing line '{line}': {e}")
                
                # Be nice to the API rate limits
                if i + batch_size < len(spanish_words):
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Error with API call: {e}")
                continue
    
    print(f"Processing complete! Enhanced data saved to: {output_file}")
    print("You can now convert this CSV to an Anki deck using the previous script.")
    
    return True

if __name__ == '__main__':
    # Get the directory of the current script
    script_dir = Path(__file__).parent.absolute()
    
    # Look for text files in the same directory
    text_files = list(script_dir.glob('*.txt'))
    
    if not text_files:
        print("No text files found in the current directory.")
    else:
        if len(text_files) == 1:
            txt_file = text_files[0]
            print(f"Found text file: {txt_file.name}")
            process_spanish_words(str(txt_file))
        else:
            print("Multiple text files found. Please select one:")
            for i, file in enumerate(text_files):
                print(f"{i+1}. {file.name}")
            
            while True:
                try:
                    choice = int(input("Enter the number of the file to process: "))
                    if 1 <= choice <= len(text_files):
                        txt_file = text_files[choice-1]
                        print(f"Processing {txt_file.name}...")
                        process_spanish_words(str(txt_file))
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(text_files)}")
                except ValueError:
                    print("Please enter a valid number")
