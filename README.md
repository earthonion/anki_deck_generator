# Anki Deck Generator

This tool was originally created for learning Spanish but can be easily adapted for other languages.

## Overview

It automates the creation of rich Anki decks with:

- Word translations  
- Example sentences  
- Audio pronunciation  
- AI-generated images  

## Usage

1. **Download Frequency List**  
   Run `dl_freq.py` to download a frequency dictionary from Wiktionary.  
   You can update the script or URL to customize the word list.
   you can browse frequency lists here:
   https://en.m.wiktionary.org/wiki/User:Matthias_Buchmeier

3. **Generate Translations**  
   Run `mk_ai_deck.py` to create a CSV with:
   - English translations
   - Example sentences

4. **Generate Media & Anki Deck**  
   Run `gen.py` to:
   - Generate audio (word and examples)  
   - Generate images for each word (requires OpenAI API access)  
   - Compile everything into an `.apkg` file

   **Note:**  
   - You’ll need a valid OpenAI API key.  
   - Image generation (especially using DALL·E 3) requires account approval.  
   - **This step can become expensive—use with caution.**

## Prebuilt Decks

Prebuilt `.apkg` decks are available in the [Releases](../../releases) section.
