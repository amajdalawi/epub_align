from ebooklib import epub
from bs4 import BeautifulSoup
import re
from typing import List, Tuple, Dict
import ebooklib

import spacy
from spacy.lang.fr import French
from spacy.lang.en import English
from pprint import pprint

class Section:

  def __init__(self, full_text: str, original_language: str):
    self.full_text = full_text
    self.original_language = original_language
    if (original_language == "en"):
      self.nlp = English()
    else:
      self.nlp = French()
    self.list_of_sents: list[list[str]] = []
    self._split_sents()
    self.proper_representation_list_of_sents()


  def _split_sents(self):
    doc = self.nlp(self.full_text)
    list_of_sents = []
    current_sent = []
    for token in doc:
      if not token.is_punct:
        try:
          if current_sent[-1].is_left_punct:
            current_sent.append(token)
          elif current_sent[-1].is_punct:
            list_of_sents.append(current_sent)
            current_sent = [token]
          else:
            current_sent.append(token)
        except Exception as e:
          current_sent.append(token)
      else:
        if token.is_right_punct:
          current_sent.append(token)
        elif token.is_left_punct:
          list_of_sents.append(current_sent)
          current_sent = [token]
        else:
          current_sent.append(token)
    self.list_of_sents = list_of_sents

  def proper_representation_list_of_sents(self):
    # for french
    if self.original_language == "fr" or self.original_language == "en":
      list_of_str = []
      for i in self.list_of_sents:
        sent = ""
        for token in i:
          if token.text[-1] == "’":
            sent += token.text
          # elif token.text[0]=="-":
          #   sent = sent[:-1] + token.text + " "
          else:
            sent += token.text + " "
        # print(sent)
        list_of_str.append(sent.strip())
      self.list_of_str = list_of_str
    else:
      # for english.
      ...


def write_file_new(text, lang, name_of_file):
  with open(f"{name_of_file}","w",encoding="utf-8") as f:
    for line in text:
      f.write(line + "\n")


def overlap_text(text_file_name): 
  try:
      print(f"trying to save .... overlapped_{text_file_name}")
      command = [
          "./vecalign/overlap.py",
          "-i", text_file_name,
          "-o", f"overlapped_{text_file_name}",
          "-n", "10"
      ]
      process = subprocess.Popen(command)
      stdout, stderr = process.communicate()
    #   if process.returncode != 0:
    #       print(f"Error during alignment: {stderr}")
    #       return None
      return f"overlapped_{text_file_name}"
  except FileNotFoundError:
      print("Error: vecalign executable not found. Make sure vecalign is installed and in the correct path.")
      return None
  except Exception as e:
      print(f"An unexpected error occurred: {e}")
      return None

from sentence_transformers import SentenceTransformer, util
import numpy as np

def encode_text_file(input_file, output_file, model_name='distiluse-base-multilingual-cased-v1'):
    """
    Encodes a text file into embeddings using a multilingual Sentence Transformer model.

    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to the output binary file for embeddings.
        model_name (str): Name of the Sentence Transformer model to use.
                           Defaults to 'all-mpnet-base-v2'.
    """
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error loading Sentence Transformer model: {e}")
        return

    sentences = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            sentences.append(line.strip())
    
    try:
        embeddings = model.encode(sentences, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy().astype(np.float32) # Ensure float32 and move to CPU

        with open(output_file, 'wb') as outfile:
            embeddings.tofile(outfile)
        print(f"Embeddings saved to {output_file}")
    except Exception as e:
        print(f"Error generating or saving embeddings: {e}")

import re

# def extract_aligned_sentences(output_string):
#     """
#     Extracts source and target sentences from the output string of vecalign.

#     Args:
#         output_string: The output string from vecalign.

#     Returns:
#         A list of tuples, where each tuple contains the source and target sentences.
#     """
#     pattern = r"SRC: (.*?)\n\s*TGT: (.*?)\n"
#     matches = re.findall(pattern, output_string, re.DOTALL)
#     aligned_sentences = [(src.strip(), tgt.strip()) for src, tgt in matches]
#     return aligned_sentences




# prompt: !./vecalign/vecalign.py --alignment_max_size 6 --src ./some_text_fr.txt --tgt some_text_en.txt --src_embed overlapped_fr.txt some_text_fr.bin --tgt_embed overlapped_en.txt some_text_en.bin --print_aligned_text -v
# turn this into python function, with arguments for, src file, overlapped src file, and embedded src file (binary), same for the three tgt files, and then the alignment_max

import subprocess
import re

def align_texts(src_file, overlapped_src_file, embedded_src_file, tgt_file, overlapped_tgt_file, embedded_tgt_file, alignment_max_size):
    """
    Aligns two text files using vecalign.

    Args:
        src_file (str): Path to the source text file.
        overlapped_src_file (str): Path to the overlapped source text file.
        embedded_src_file (str): Path to the embedded source binary file.
        tgt_file (str): Path to the target text file.
        overlapped_tgt_file (str): Path to the overlapped target text file.
        embedded_tgt_file (str): Path to the embedded target binary file.
        alignment_max_size (int): Maximum alignment size.

    Returns:
        list: A list of tuples, where each tuple contains the source and target sentences.
              Returns None if the alignment process fails.
    """
    try:
        command = [
            "./vecalign/vecalign.py",
            "--alignment_max_size", str(alignment_max_size),
            "--src", src_file,
            "--tgt", tgt_file,
            "--src_embed", overlapped_src_file, embedded_src_file,
            "--tgt_embed", overlapped_tgt_file, embedded_tgt_file,
            "--print_aligned_text", "-v"
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error during alignment: {stderr}")
            return None

        return extract_aligned_sentences(stdout)
    except FileNotFoundError:
        print("Error: vecalign executable not found. Make sure vecalign is installed and in the correct path.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def extract_aligned_sentences(output_string):
    """
    Extracts source and target sentences from the output string of vecalign.
    """
    pattern = r"SRC: (.*?)\n\s*TGT: (.*?)\n"
    matches = re.findall(pattern, output_string, re.DOTALL)
    aligned_sentences = [(src.strip(), tgt.strip()) for src, tgt in matches]
    return aligned_sentences



class Book:
    def __init__(self, path_lang1: str, path_lang2: str):
        self.path_lang1 = path_lang1
        self.path_lang2 = path_lang2

        self.sections_lang1: Dict[str, str] = self._extract_sections(path_lang1)
        self.sections_lang2: Dict[str, str] = self._extract_sections(path_lang2)

        self.aligned_sections: List[Tuple[str, str, str, str]] = []
        # self.unaligned_sections_lang2: Dict[str, str] = self.sections_lang2.copy()
        self.align_sections()
        self._sectionify()
        self.final_aligned_sentences: List[Tuple[str, str, str, str]] = []
        self._align_sentences_final()


    def _extract_sections(self, epub_path: str) -> Dict[str, str]:
        book = epub.read_epub(epub_path)
        sections = {}

        count = 0
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                if len(text) > 100:
                    sections[count] = text
                    count +=1
        return sections

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        tokens1 = set(re.findall(r'\w+', text1.lower()))
        tokens2 = set(re.findall(r'\w+', text2.lower()))
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union)

    def align_sections(self):
        print('alright here we go')
        for id1, text1 in self.sections_lang1.items():
            best_match_id = None
            best_score = 0.0

            for id2, text2 in self.sections_lang2.items():
                score = self._jaccard_similarity(text1, text2)
                if score > best_score:
                    best_score = score
                    best_match_id = id2

            if best_match_id is not None:
                # text2 = self.unaligned_sections_lang2.pop(best_match_id)
                text2 = self.sections_lang2[best_match_id]
                self.aligned_sections.append((id1, best_match_id, text1, text2))
            else:
                self.aligned_sections.append((id1, None, text1, ""))  # No match found

    def show_aligned_sections(self, limit: int = 5):
        print(f"\n--- Showing {limit} aligned sections ---")
        for i, (id1, id2, text1, text2) in enumerate(self.aligned_sections[:limit]):
            print(f"\n[Aligned #{i+1}]")
            print(f"[Lang1 ID: {id1}]: {text1[:300]}...")
            print(f"[Lang2 ID: {id2 if id2 else 'None'}]: {text2[:300]}...")

    def show_unaligned_sections_lang2(self, limit: int = 5):
        print(f"\n--- Remaining unaligned Lang2 Sections ---")
        for i, (id2, text2) in enumerate(list(self.unaligned_sections_lang2.items())[:limit]):
            print(f"[Lang2 #{i+1} - {id2}]: {text2[:300]}...")

    def _sectionify(self):
        self.sections = []
        for tu in self.aligned_sections:
            s_fr = Section(tu[3], "fr")
            s_en = Section(tu[2], "en")
            self.sections.append((s_fr, s_en))
    
    def _align_sentences_final(self):
        # self.final_aligned_sentences = []
        print(len(self.sections))
        assert len(self.sections) > 10
        for i, section in enumerate(self.sections):
            section_fr_file_name = f"section_{i}_fr.txt"
            section_en_file_name = f"section_{i}_en.txt"
            print(section[0].list_of_str)
            print(section[1].list_of_str)
            write_file_new(section[0].list_of_str, "fr", section_fr_file_name)
            write_file_new(section[1].list_of_str, "en", section_en_file_name)
            overlap_text(section_fr_file_name)
            overlap_text(section_en_file_name)
            encode_text_file(f"overlapped_{section_fr_file_name}",f"overlapped_{section_fr_file_name}.bin")
            encode_text_file(f"overlapped_{section_en_file_name}",f"overlapped_{section_en_file_name}.bin")
            aligned_sentences = align_texts(section_fr_file_name,f"overlapped_{section_fr_file_name}",f"overlapped_{section_fr_file_name}.bin",
                                            section_en_file_name,f"overlapped_{section_en_file_name}",f"overlapped_{section_en_file_name}.bin", 8)
            pprint(aligned_sentences)
            self.final_aligned_sentences.append(aligned_sentences)

    
if __name__ == "__main__":
    book = Book("en.epub", "fr.epub")
    book.align_sections()
    # book.align_sections(similarity_threshold=70)
    # # book.print_aligned()
    # book.show_aligned_sections(limit=30)
    # print(book.aligned_sections[5][2])

    book._align_sentences_final()
    for i in book.final_aligned_sentences:
      print(i)

    # print(book.aligned_sections)

    # book.show_aligned_sections()
    # book.show_unaligned_sections(limit=3)