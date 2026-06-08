"""PDF parsing and structure extraction"""

import fitz
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Section:
    """Represents a section in the paper"""
    name: str
    start_page: int
    end_page: int
    start_line: int = 0
    end_line: int = 0
    content: str = ""
    
@dataclass
class ParsedDocument:
    """Complete parsed document structure"""
    file_path: str
    title: str
    abstract: str
    sections: Dict[str, Section]
    full_text: str
    num_pages: int
    
class PDFParser:
    """Parser for extracting structure from academic PDFs"""
    
    def __init__(self):
        self.section_headers = [
            "abstract", "introduction", "related work", "background",
            "methodology", "methods", "experimental setup", "experiments",
            "results", "discussion", "conclusion", "references",
            "acknowledgements", "appendix", "supplementary"
        ]
        
        # Section name normalization mapping
        self.section_mapping = {
            "abstract": "Abstract",
            "introduction": "Introduction",
            "related work": "Related Work",
            "background": "Background",
            "methodology": "Methodology",
            "methods": "Methodology",
            "experimental setup": "Methodology",
            "experiments": "Experiments",
            "results": "Results",
            "discussion": "Discussion",
            "conclusion": "Conclusion",
            "references": "References",
        }
    
    def parse(self, pdf_path: str) -> ParsedDocument:
        """Parse a PDF file and extract its structure"""
        
        doc = fitz.open(pdf_path)
        
        # Extract title and abstract
        title, abstract = self._extract_title_abstract(doc)
        
        # Identify sections
        sections = self._identify_sections(doc)
        
        # Extract full text
        full_text = self._extract_full_text(doc)
        
        doc.close()
        
        return ParsedDocument(
            file_path=pdf_path,
            title=title,
            abstract=abstract,
            sections=sections,
            full_text=full_text,
            num_pages=len(doc)
        )
    
    def _extract_title_abstract(self, doc) -> Tuple[str, str]:
        """Extract paper title and abstract from first page"""
        first_page = doc[0].get_text()
        lines = first_page.split('\n')
        
        title = ""
        abstract = ""
        in_abstract = False
        abstract_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Detect abstract section
            if 'abstract' in line.lower() and len(line) < 30:
                in_abstract = True
                continue
            elif in_abstract:
                # Check for next section header
                if any(header in line.lower() for header in ['introduction', 'keywords']) and len(line) < 40:
                    break
                if len(abstract_lines) < 30:  # Limit abstract length
                    abstract_lines.append(line)
            
            # Get title (first few lines before abstract)
            elif not title and not in_abstract and len(line) > 10:
                if not any(stop in line.lower() for stop in ['abstract', 'keywords', 'doi']):
                    title = line
                    # Continue to next line if title might be multi-line
                    if i + 1 < len(lines) and len(lines[i+1].strip()) > 10:
                        title += " " + lines[i+1].strip()
        
        abstract = ' '.join(abstract_lines)
        
        # Clean title
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title, abstract
    
    def _identify_sections(self, doc) -> Dict[str, Section]:
        """Identify section boundaries in the document"""
        
        section_starts = {}
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            lines = text.split('\n')
            
            for line_num, line in enumerate(lines):
                line_lower = line.lower().strip()
                
                # Check if line matches a section header
                for header in self.section_headers:
                    # Match exact or near-exact header
                    if line_lower == header or (header in line_lower and len(line) < 60):
                        if header not in section_starts:
                            section_starts[header] = {
                                'page': page_num + 1,
                                'line': line_num,
                                'text': line.strip()
                            }
                        break
        
        # Build section boundaries
        sections = {}
        sorted_sections = sorted(section_starts.items(), key=lambda x: (x[1]['page'], x[1]['line']))
        
        for i, (section_name, start_info) in enumerate(sorted_sections):
            # Determine end page
            if i + 1 < len(sorted_sections):
                end_page = sorted_sections[i + 1][1]['page'] - 1
                if end_page < start_info['page']:
                    end_page = start_info['page']
            else:
                end_page = None  # Until end of document
            
            # Get normalized section name
            normalized_name = self.section_mapping.get(section_name, section_name.title())
            
            sections[normalized_name] = Section(
                name=normalized_name,
                start_page=start_info['page'],
                end_page=end_page if end_page else 999,
                start_line=start_info['line'],
                content=""  # Will be filled if needed
            )
        
        return sections
    
    def _extract_full_text(self, doc) -> str:
        """Extract full text from all pages"""
        full_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            full_text.append(text)
        
        return '\n'.join(full_text)
    
    def get_section_for_page(self, page_num: int, sections: Dict[str, Section]) -> str:
        """Get the section name for a given page number"""
        for section_name, section in sections.items():
            if section.start_page <= page_num <= section.end_page:
                return section_name
        return "Unknown"
    
    def get_section_content(self, doc, section: Section) -> str:
        """Extract content for a specific section"""
        content_lines = []
        
        for page_num in range(section.start_page - 1, section.end_page):
            page = doc[page_num]
            text = page.get_text()
            
            if page_num == section.start_page - 1 and section.start_line > 0:
                # Start from specific line
                lines = text.split('\n')
                content_lines.extend(lines[section.start_line:])
            else:
                content_lines.append(text)
            
            if page_num + 1 == section.end_page:
                # Stop at section end
                break
        
        return '\n'.join(content_lines)