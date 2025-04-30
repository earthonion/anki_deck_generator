#!/usr/bin/env python3
"""
Spanish Audio & Image Flashcards Generator
With support for local image generation and word-based file naming
"""

import csv
import os
import time
import requests
import argparse
import tempfile
from pathlib import Path
from gtts import gTTS
import genanki
import base64
import json
import random

def generate_audio_and_images_for_spanish(input_file, openai_api_key=None, org_id=None, use_local_gen=False):
    """
    Generate audio files and images for Spanish words and sentences from a CSV file.
    Skip generation if files already exist.
    
    Parameters:
    - input_file: Path to the CSV file with Spanish words and sentences
    - openai_api_key: OpenAI API key for TTS and image generation
    - org_id: Optional OpenAI organization ID
    - use_local_gen: Whether to use local image generation
    """
    input_path = Path(input_file)
    output_file = str(input_path.with_name(f"{input_path.stem}.apkg"))
    media_dir = input_path.parent / "anki_media"
    
    # Create media directory if it doesn't exist
    media_dir.mkdir(exist_ok=True)
    
    # Define column names we're looking for
    col_names = {
        'spanish': ['spanish', 'front'],
        'english': ['english', 'back'],
        'spanish_example': ['spanishexample', 'spanish example'],
        'english_example': ['englishexample', 'english example']
    }
    
    # Read the CSV data
    data_rows = []
    field_mapping = {}
    
    try:
        with open(input_file, 'r', encoding='utf-8') as csvfile:
            sample = csvfile.read(1024)
            csvfile.seek(0)
            
            # Detect CSV dialect and headers
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)
            
            if has_header:
                reader = csv.DictReader(csvfile, dialect=dialect)
                headers = reader.fieldnames
                
                # Map field names
                for header in headers:
                    normalized = header.lower().strip().replace(' ', '')
                    for category, options in col_names.items():
                        if normalized in options:
                            field_mapping[category] = header
            else:
                reader = csv.reader(csvfile, dialect=dialect)
                # Assume default column order if no header
                field_mapping = {
                    'spanish': 0,
                    'english': 1,
                    'spanish_example': 2,
                    'english_example': 3
                }
            
            # Reset file pointer
            csvfile.seek(0)
            if has_header:
                next(reader)
                
            # Read data rows
            for row in reader:
                if isinstance(row, dict):
                    data = {
                        'spanish': row.get(field_mapping.get('spanish', 'Spanish'), '').strip().replace('"', ''),
                        'english': row.get(field_mapping.get('english', 'English'), '').strip().replace('"', ''),
                        'spanish_example': row.get(field_mapping.get('spanish_example', 'SpanishExample'), '').strip().replace('"', ''),
                        'english_example': row.get(field_mapping.get('english_example', 'EnglishExample'), '').strip().replace('"', '')
                    }
                else:  # List row format
                    spanish_idx = field_mapping.get('spanish', 0)
                    english_idx = field_mapping.get('english', 1)
                    spanish_example_idx = field_mapping.get('spanish_example', 2)
                    english_example_idx = field_mapping.get('english_example', 3)
                    
                    data = {
                        'spanish': row[spanish_idx].strip().replace('"', '') if len(row) > spanish_idx else '',
                        'english': row[english_idx].strip().replace('"', '') if len(row) > english_idx else '',
                        'spanish_example': row[spanish_example_idx].strip().replace('"', '') if len(row) > spanish_example_idx else '',
                        'english_example': row[english_example_idx].strip().replace('"', '') if len(row) > english_example_idx else ''
                    }
                
                if data['spanish'] and data['english']:
                    data_rows.append(data)
    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False
    
    print(f"Found {len(data_rows)} Spanish words and phrases to process")
    
    # Import local image generator if using local generation
    local_generator = None
    if use_local_gen:
        try:
            from local_img import BitsAndBytesImageGenerator
            local_generator = BitsAndBytesImageGenerator(output_dir=str(media_dir))
            print("Using local image generator")
        except ImportError as e:
            print(f"Error importing local image generator: {e}")
            print("Falling back to OpenAI for image generation")
            use_local_gen = False
    
    # Generate audio files and images
    audio_files = {}
    image_files = {}
    count = 0
    total = len(data_rows)
    for data in data_rows:
        count += 1
        spanish_word = data['spanish']
        print(f"Processing: {spanish_word}")
        print(f"Completed: {count}/{total}")
        
        # Use sanitized word for filenames
        word_safe = sanitize_filename(spanish_word)
        
        # 1. Generate audio for Spanish word
        word_filename = f"word_{word_safe}.mp3"
        word_path = media_dir / word_filename
        
        # Check if audio file already exists
        if word_path.exists():
            print(f"  Audio for word already exists, skipping generation")
            audio_files[spanish_word] = word_filename
        else:
            try:
                if openai_api_key:
                    # Use OpenAI TTS API if key is provided
                    generate_openai_tts(spanish_word, word_path, openai_api_key)
                else:
                    # Fall back to gTTS
                    generate_gtts(spanish_word, word_path, lang='es')
                    
                audio_files[spanish_word] = word_filename
                print(f"  Word audio generated successfully")
            except Exception as e:
                print(f"  Error generating audio for word '{spanish_word}': {e}")
        
        # 2. Generate audio for Spanish example if available
        if data['spanish_example']:
            # Use sanitized word for consistency, not the whole example
            example_filename = f"example_{word_safe}.mp3"
            example_path = media_dir / example_filename
            
            # Check if example audio file already exists
            if example_path.exists():
                print(f"  Audio for example sentence already exists, skipping generation")
                audio_files[data['spanish_example']] = example_filename
            else:
                try:
                    if openai_api_key:
                        generate_openai_tts(data['spanish_example'], example_path, openai_api_key)
                    else:
                        generate_gtts(data['spanish_example'], example_path, lang='es')
                        
                    audio_files[data['spanish_example']] = example_filename
                    print(f"  Example audio generated successfully")
                except Exception as e:
                    print(f"  Error generating audio for example '{data['spanish_example']}': {e}")
        
        # 3. Generate image for the word
        image_filename = f"anki_{word_safe}.jpg"
        image_path = media_dir / image_filename
        
        # Check if image already exists
        if image_path.exists():
            print(f"  Image already exists, skipping generation")
            image_files[spanish_word] = image_filename
        elif use_local_gen and local_generator:
            # Use local image generator
            try:
                # Generate image prompt
                prompt = local_generator.generate_image_prompt(spanish_word, data['english'])
                print(f"  Image prompt: {prompt}")
                
                # Generate the image using local generator
                local_generator.generate_image(prompt, image_path)
                image_files[spanish_word] = image_filename
                print(f"  Image generated successfully using local generator")
            except Exception as e:
                print(f"  Error generating image locally for '{spanish_word}': {e}")
        elif openai_api_key:
            # Use OpenAI for image generation
            try:
                # Generate image prompt using the word and its definition
                prompt = generate_image_prompt(spanish_word, data['english'], openai_api_key, org_id)
                print(f"  Image prompt: {prompt}")
                
                # Generate the image using OpenAI
                generate_image(prompt, image_path, openai_api_key, org_id)
                image_files[spanish_word] = image_filename
                print(f"  Image generated successfully with OpenAI")
            except Exception as e:
                print(f"  Error generating image with OpenAI for '{spanish_word}': {e}")
        
        # Add a small delay to avoid rate limits
        #time.sleep(0.5)
    
    # Clean up local generator if used
    if use_local_gen and local_generator:
        try:
            local_generator.close()
            print("Local image generator closed")
        except:
            pass
    
    # Create Anki deck with audio and images
    try:
        create_anki_deck_with_audio_and_images(data_rows, audio_files, image_files, output_file, media_dir)
        print(f"Created Anki package at {output_file}")
        return True
    except Exception as e:
        print(f"Error creating Anki deck: {e}")
        return False

def generate_openai_tts(text, output_path, api_key):
    """Generate audio using OpenAI's TTS API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": "alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
        "response_format": "mp3"
    }
    
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Error from OpenAI API: {response.text}")
        raise Exception(f"OpenAI API error: {response.status_code}")

def generate_gtts(text, output_path, lang='es'):
    """Generate audio using Google Text-to-Speech"""
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(str(output_path))

def generate_image_prompt(spanish_word, english_translation, api_key, org_id=None):
    """
    Generate an optimal prompt for image generation using GPT
    
    Parameters:
    - spanish_word: The Spanish word
    - english_translation: The English translation
    - api_key: OpenAI API key
    - org_id: Optional OpenAI organization ID
    """
    try:
        from openai import OpenAI
        
        # Initialize OpenAI client with organization ID if provided
        if org_id:
            client = OpenAI(api_key=api_key, organization=org_id)
        else:
            client = OpenAI(api_key=api_key)
        
        prompt = f"""Create a clear, simple image prompt for the Spanish word "{spanish_word}" which means "{english_translation}" in English. 
        The prompt should describe a simple, memorable image that illustrates the meaning of the word.
        Make it suitable for language learning flashcards.
        Write only the prompt text, nothing else."""
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Or use gpt-3.5-turbo for lower cost
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates optimal image generation prompts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        image_prompt = response.choices[0].message.content.strip()
        return image_prompt
    
    except Exception as e:
        print(f"Error generating image prompt: {e}")
        return f"A simple illustration of the concept: {english_translation}"

def generate_image(prompt, output_path, api_key, org_id=None):
    """
    Generate an image using OpenAI's GPT-Image-1 API
    Based on OpenAI Cookbook examples
    
    Parameters:
    - prompt: The text prompt for image generation
    - output_path: Where to save the generated image
    - api_key: OpenAI API key
    - org_id: Optional organization ID
    """
    from openai import OpenAI
    import base64
    from io import BytesIO
    
    # Initialize OpenAI client with organization ID if provided
    if org_id:
        client = OpenAI(api_key=api_key, organization=org_id)
    else:
        client = OpenAI(api_key=api_key)
    
    try:
        # Generate image using GPT-Image-1
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            quality="high",
            size="1024x1024",
            # Note: response_format parameter is not used with gpt-image-1
            # It returns b64_json by default
        )
        
        # Extract base64 data from response
        image_base64 = result.data[0].b64_json
        
        # Decode and save image
        image_bytes = base64.b64decode(image_base64)
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
        
        print(f"  Image saved successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"  Error generating image: {e}")
        return False

def sanitize_filename(filename):
    """Create a valid filename from text"""
    # Replace invalid characters
    valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    sanitized = ''.join(c for c in filename if c in valid_chars)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Limit length and trim
    return sanitized[:50].strip()

def create_anki_deck_with_audio_and_images(data_rows, audio_files, image_files, output_file, media_dir):
    """Create an Anki deck with audio files and images"""
    # Define model
    model_id = random.randrange(1 << 30, 1 << 31)
    
    model = genanki.Model(
        model_id,
        'Spanish Audio & Image Model',
        fields=[
            {'name': 'Spanish'},
            {'name': 'English'},
            {'name': 'SpanishExample'},
            {'name': 'EnglishExample'},
            {'name': 'WordAudio'},
            {'name': 'ExampleAudio'},
            {'name': 'Image'}
        ],
        templates=[
            {
                'name': 'Spanish to English',
                'qfmt': '''
                {{WordAudio}}
                <div style="font-size: 28px; text-align: center; color: #2563eb; font-weight: bold; margin-bottom: 15px;">{{Spanish}}</div>
                {{#Image}}<div style="text-align: center; margin-top: 15px; margin-bottom: 15px;"><img src="{{Image}}" style="max-width: 300px; max-height: 300px;"></div>{{/Image}}
                ''',
                'afmt': '''
                {{WordAudio}}
                <div style="font-size: 28px; text-align: center; color: #2563eb; font-weight: bold; margin-bottom: 15px;">{{Spanish}}</div>
                {{#Image}}<div style="text-align: center; margin-top: 15px; margin-bottom: 15px;"><img src="{{Image}}" style="max-width: 300px; max-height: 300px;"></div>{{/Image}}
                <hr id="answer">
                <div style="font-size: 24px; text-align: center; color: #059669;">{{English}}</div>
                {{#SpanishExample}}
                <div style="margin-top: 20px; font-size: 20px; text-align: center; color: #6366f1; font-style: italic;">
                    {{ExampleAudio}}{{SpanishExample}}
                </div>
                <div style="font-size: 18px; text-align: center; color: #4b5563;">{{EnglishExample}}</div>
                {{/SpanishExample}}
                '''
            },
            {
                'name': 'English to Spanish',
                'qfmt': '''
                <div style="font-size: 24px; text-align: center; color: #059669;">{{English}}</div>
                ''',
                'afmt': '''
                <div style="font-size: 24px; text-align: center; color: #059669;">{{English}}</div>
                <hr id="answer">
                {{WordAudio}}
                <div style="font-size: 28px; text-align: center; color: #2563eb; font-weight: bold; margin-bottom: 15px;">{{Spanish}}</div>
                {{#Image}}<div style="text-align: center; margin-top: 15px; margin-bottom: 15px;"><img src="{{Image}}" style="max-width: 300px; max-height: 300px;"></div>{{/Image}}
                {{#SpanishExample}}
                <div style="margin-top: 20px; font-size: 20px; text-align: center; color: #6366f1; font-style: italic;">
                    {{ExampleAudio}}{{SpanishExample}}
                </div>
                <div style="font-size: 18px; text-align: center; color: #4b5563;">{{EnglishExample}}</div>
                {{/SpanishExample}}
                '''
            }
        ],
        css='''
        .card {
            font-family: Arial, sans-serif;
            font-size: 20px;
            text-align: center;
            color: black;
            background-color: white;
            padding: 20px;
        }
        img {
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        '''
    )
    
    # Create deck
    deck_id = random.randrange(1 << 30, 1 << 31)
    deck_name = Path(output_file).stem.replace('_', ' ').title()
    deck = genanki.Deck(deck_id, f"Spanish Audio & Images: {deck_name}")
    
    # Add media files
    media_files = []
    for filepath in media_dir.glob("*.*"):
        if filepath.suffix.lower() in ['.mp3', '.jpg', '.png']:
            media_files.append(str(filepath))
    
    # Create package with media
    package = genanki.Package(deck)
    package.media_files = media_files
    
    # Add notes with audio and images
    for data in data_rows:
        spanish_word = data['spanish']
        
        # Get audio filenames if they exist
        word_audio = audio_files.get(spanish_word, '')
        example_audio = audio_files.get(data['spanish_example'], '')
        image = image_files.get(spanish_word, '')
        
        # Format audio tags for Anki
        word_audio_tag = f'[sound:{word_audio}]' if word_audio else ''
        example_audio_tag = f'[sound:{example_audio}]' if example_audio else ''
        
        # Create note
        note = genanki.Note(
            model=model,
            fields=[
                spanish_word,
                data['english'],
                data['spanish_example'],
                data['english_example'],
                word_audio_tag,
                example_audio_tag,
                image
            ]
        )
        
        deck.add_note(note)
    
    # Write package to file
    package.write_to_file(output_file)

if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Spanish Audio & Image Flashcards")
    parser.add_argument("input_file", nargs="?", help="Path to the CSV file with Spanish words and phrases")
    parser.add_argument("--api-key", help="OpenAI API key for audio and image generation")
    parser.add_argument("--org-id", help="OpenAI organization ID (if required)")
    parser.add_argument("--local", action="store_true", help="Use local image generation instead of OpenAI")
    
    args = parser.parse_args()
    
    # Get the input file
    input_file = args.input_file
    
    if not input_file:
        # Look for CSV files in the current directory
        csv_files = list(Path().glob('*.csv'))
        
        if not csv_files:
            print("No CSV files found in the current directory.")
            print("Please specify a CSV file or place one in the current directory.")
            exit(1)
        
        if len(csv_files) == 1:
            input_file = str(csv_files[0])
            print(f"Found CSV file: {input_file}")
        else:
            print("Multiple CSV files found. Please select one:")
            for i, file in enumerate(csv_files):
                print(f"{i+1}. {file.name}")
            
            while True:
                try:
                    choice = int(input("Enter the number of the file to process: "))
                    if 1 <= choice <= len(csv_files):
                        input_file = str(csv_files[choice-1])
                        print(f"Selected: {input_file}")
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(csv_files)}")
                except ValueError:
                    print("Please enter a valid number")
    
    # Get API key if not provided
    api_key = args.api_key
    
    if not api_key and not args.local:
        # Try to get from environment variable
        api_key = "" #your openai api key
        
        if not api_key:
            use_api = input("Use OpenAI for audio and image generation? (y/n): ").strip().lower() == 'y'
            if use_api:
                api_key = input("Enter your OpenAI API key: ").strip()
                if not api_key:
                    print("No API key provided. Will use Google TTS for audio and local generation for images.")
    
    # Get organization ID if not provided
    org_id = args.org_id
    
    if api_key and not org_id:
        use_org_id = input("Do you need to specify an organization ID? (y/n): ").strip().lower() == 'y'
        if use_org_id:
            org_id = input("Enter your OpenAI organization ID: ").strip()
    
    # Generate the flashcards
    print(f"Processing file: {input_file}")
    print(f"Using local image generation: {'Yes' if args.local else 'No'}")
    generate_audio_and_images_for_spanish(input_file, api_key, org_id, args.local)
