from ebooklib import epub
from bs4 import BeautifulSoup
import re
from typing import List, Tuple, Dict
import ebooklib





class Book:
    def __init__(self, path_lang1: str, path_lang2: str):
        self.path_lang1 = path_lang1
        self.path_lang2 = path_lang2

        self.sections_lang1: Dict[str, str] = self._extract_sections(path_lang1)
        self.sections_lang2: Dict[str, str] = self._extract_sections(path_lang2)

        self.aligned_sections: List[Tuple[str, str, str, str]] = []
        self.unaligned_sections_lang2: Dict[str, str] = self.sections_lang2.copy()

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
    
if __name__ == "__main__":
    book = Book("en.epub", "fr.epub")
    book.align_sections()
    # book.align_sections(similarity_threshold=70)
    # book.print_aligned()
    book.show_aligned_sections(limit=30)
    print(book.aligned_sections[5][2])

    # print(book.aligned_sections)

    # book.show_aligned_sections()
    # book.show_unaligned_sections(limit=3)