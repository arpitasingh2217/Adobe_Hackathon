import fitz  # PyMuPDF
import json
import os
import re
from collections import defaultdict
import sys
import logging

# Setup minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pdf_extractor')

def calculate_visual_gap_features(text_blocks_by_page):
    """
    Calculate visual spacing features to identify heading patterns
    Returns a dictionary of page indices with spacing metrics
    """
    spacing_features = {}
    
    for page_num, blocks in enumerate(text_blocks_by_page):
        if not blocks or len(blocks) < 2:
            continue
            
        # Sort blocks by vertical position
        sorted_blocks = sorted(blocks, key=lambda b: b["y0"])
        
        # Calculate gaps between blocks
        gaps = []
        for i in range(1, len(sorted_blocks)):
            prev_block = sorted_blocks[i-1]
            curr_block = sorted_blocks[i]
            gap = curr_block["y0"] - prev_block["y1"]
            gaps.append((i-1, i, gap))
        
        # Identify significant gaps (potential heading separators)
        if gaps:
            avg_gap = sum(g[2] for g in gaps) / len(gaps)
            std_gap = (sum((g[2] - avg_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5
            
            # Gaps larger than average + 0.5*std are potentially significant
            significant_gaps = [(i, j) for i, j, gap in gaps if gap > avg_gap + 0.5 * std_gap]
            
            spacing_features[page_num] = {
                "avg_gap": avg_gap,
                "std_gap": std_gap,
                "significant_gaps": significant_gaps
            }
    
    return spacing_features

def analyze_text_linguistic_features(text):
    """
    Analyze linguistic features to determine if text is likely a heading
    """
    # Clean text and lowercase for analysis
    clean_text = re.sub(r'\s+', ' ', text).strip()
    lower_text = clean_text.lower()
    
    # Features that suggest heading
    heading_indicators = {
        "ends_with_colon": clean_text.endswith(':'),
        "all_caps": clean_text.upper() == clean_text and len(clean_text) > 3,
        "title_case": all(w[0].isupper() for w in clean_text.split() if len(w) > 3),
        "starts_with_number": bool(re.match(r'^\d+', clean_text)),
        "numbered_section": bool(re.match(r'^\d+\.\d+(\.\d+)*\s', clean_text)),
        "short_text": len(clean_text.split()) <= 6,
        "starts_with_heading_word": any(lower_text.startswith(w) for w in [
            'chapter', 'section', 'introduction', 'summary', 'conclusion', 'overview', 'appendix'
        ])
    }
    
    # Features that suggest paragraph text
    paragraph_indicators = {
        "starts_with_lowercase": len(clean_text) > 1 and clean_text[0].islower(),
        "contains_stop_words": any(f" {w} " in f" {lower_text} " for w in [
            'and', 'the', 'of', 'to', 'a', 'in', 'for', 'with', 'on', 'at'
        ]),
        "ends_with_period": clean_text.endswith('.'),
        "long_text": len(clean_text.split()) > 10,
        "contains_sentence_connectors": any(w in lower_text for w in [
            ' because ', ' therefore ', ' however ', ' thus ', ' moreover ', ' furthermore '
        ])
    }
    
    # Calculate scores
    heading_score = sum(1 for k, v in heading_indicators.items() if v)
    paragraph_score = sum(1 for k, v in paragraph_indicators.items() if v)
    
    # Normalize by number of features
    heading_score = heading_score / len(heading_indicators)
    paragraph_score = paragraph_score / len(paragraph_indicators)
    
    return {
        "heading_score": heading_score,
        "paragraph_score": paragraph_score,
        "likely_heading": heading_score > paragraph_score,
        "confidence": abs(heading_score - paragraph_score)
    }

def analyze_font_consistency(blocks):
    """
    Analyze font consistency to identify headings based on format changes
    """
    if not blocks:
        return {}
        
    # Extract font information
    fonts = [(block["size"], block.get("is_bold", False), block.get("is_italic", False)) 
             for block in blocks if "size" in block]
    
    if not fonts:
        return {}
        
    # Find most common font (body text)
    font_counts = defaultdict(int)
    for f in fonts:
        font_counts[f] += 1
    
    body_font = max(font_counts.items(), key=lambda x: x[1])[0]
    
    # Classify each font
    font_categories = {}
    for f in font_counts:
        size, is_bold, is_italic = f
        
        if size > body_font[0] * 1.5:
            category = "h1"
        elif size > body_font[0] * 1.2 or (size > body_font[0] * 1.05 and is_bold):
            category = "h2"
        elif is_bold or is_italic or size > body_font[0] * 1.05:
            category = "h3"
        else:
            category = "body"
            
        font_categories[f] = category
    
    return {
        "body_font": body_font,
        "font_categories": font_categories
    }

def identify_heading_sequences(headings):
    """
    Identify patterns in heading sequences to find missing or incorrect headings
    """
    if not headings:
        return []
    
    # Sort by page and position
    sorted_headings = sorted(headings, key=lambda h: (h["page"], h.get("y0", 0)))
    
    # Track heading levels and their numbering
    level_patterns = {
        "H1": [],
        "H2": [],
        "H3": []
    }
    
    # Extract patterns
    for h in sorted_headings:
        text = h["text"]
        level = h["level"]
        
        # Check for numbered patterns like "1.2.3"
        match = re.match(r'^(\d+)(?:\.(\d+))?(?:\.(\d+))?', text)
        if match:
            groups = match.groups()
            if groups[0]:  # At least one number found
                num_pattern = [int(g) for g in groups if g]
                level_patterns[level].append(num_pattern)
    
    # Check for consistency and suggest corrections
    suggestions = []
    
    # Look for gaps in numbering
    for level, patterns in level_patterns.items():
        if not patterns:
            continue
            
        # Look at first component only (e.g., the '1' in '1.2')
        first_nums = [p[0] for p in patterns if p]
        if first_nums:
            expected_range = list(range(min(first_nums), max(first_nums) + 1))
            missing = [n for n in expected_range if n not in first_nums]
            
            if missing:
                suggestions.append({
                    "level": level,
                    "issue": "missing_numbers",
                    "missing": missing
                })
    
    # Check for level consistency (e.g., H2 without parent H1)
    h1_sections = set()
    h2_parents = set()
    
    for pattern in level_patterns["H1"]:
        if pattern:
            h1_sections.add(pattern[0])
    
    for pattern in level_patterns["H2"]:
        if len(pattern) >= 1:
            h2_parents.add(pattern[0])
    
    missing_parents = [p for p in h2_parents if p not in h1_sections]
    if missing_parents:
        suggestions.append({
            "level": "H1",
            "issue": "missing_parent",
            "missing": missing_parents
        })
    
    return suggestions

def detect_outline_from_toc_page(doc, text_blocks_by_page):
    """
    Extract headings directly from table of contents page when available
    """
    extracted_toc = []
    toc_page_candidates = []
    
    # Look for pages with "contents" or "table of contents" in the first few pages
    for page_num in range(min(5, len(text_blocks_by_page))):
        page_blocks = text_blocks_by_page[page_num]
        page_text = " ".join(block["text"].lower() for block in page_blocks[:5])
        
        if "contents" in page_text or "table of content" in page_text:
            toc_page_candidates.append(page_num)
    
    if not toc_page_candidates:
        return []
        
    # Process the most likely TOC page
    toc_page = toc_page_candidates[0]
    toc_blocks = text_blocks_by_page[toc_page]
    
    # Look for TOC patterns: heading text followed by page number
    toc_pattern = re.compile(r'^(.*?)\s*\.{2,}\s*(\d+)$')
    
    current_level = "H1"
    prev_indent = None
    indent_levels = {}
    
    for i, block in enumerate(toc_blocks):
        text = block["text"].strip()
        match = toc_pattern.match(text)
        
        if match:
            heading_text = match.group(1).strip()
            page_num = int(match.group(2))
            
            # Determine level based on indentation
            indent = block.get("x0", 0)
            
            # First entry establishes baseline indent
            if prev_indent is None:
                prev_indent = indent
                indent_levels[indent] = "H1"
            # Subsequent entries establish hierarchy
            elif indent > prev_indent + 10:
                # More indented = lower level
                if current_level == "H1":
                    current_level = "H2"
                elif current_level == "H2":
                    current_level = "H3"
                indent_levels[indent] = current_level
            elif indent < prev_indent - 10:
                # Less indented = higher level
                for known_indent, level in sorted(indent_levels.items()):
                    if abs(known_indent - indent) < 10:
                        current_level = level
                        break
                    elif known_indent < indent:
                        # Find the closest but lower indent
                        if level == "H2":
                            current_level = "H2"
                        else:
                            current_level = "H1"
            
            extracted_toc.append({
                "level": current_level,
                "text": heading_text,
                "page": page_num,
                "from_toc_page": True
            })
            
            prev_indent = indent
    
    return extracted_toc

def analyze_geometric_layout(text_blocks_by_page):
    """
    Analyze geometric layout to identify heading patterns
    """
    layout_features = {}
    
    for page_num, blocks in enumerate(text_blocks_by_page):
        if not blocks:
            continue
            
        # Calculate indent distribution
        x_positions = [block["x0"] for block in blocks if "x0" in block]
        if not x_positions:
            continue
            
        # Find common indentation levels
        x_counts = defaultdict(int)
        for x in x_positions:
            # Round to nearest 5 pixels to account for slight variations
            rounded_x = round(x / 5) * 5
            x_counts[rounded_x] += 1
        
        # Sort by frequency
        common_indents = sorted([(pos, count) for pos, count in x_counts.items()], 
                              key=lambda x: x[1], reverse=True)
        
        # Identify left margin (most common x position)
        left_margin = common_indents[0][0] if common_indents else 0
        
        # Classify indentation levels
        indent_levels = {}
        for pos, count in common_indents:
            if pos <= left_margin + 5:
                indent_levels[pos] = 0  # Main text
            elif pos <= left_margin + 30:
                indent_levels[pos] = 1  # First level indent
            else:
                indent_levels[pos] = 2  # Second level indent
        
        layout_features[page_num] = {
            "left_margin": left_margin,
            "indent_levels": indent_levels
        }
    
    return layout_features

def detect_language(text):
    """
    Simple language detection based on common word frequencies
    """
    if not text or len(text) < 20:
        return "en"  # Default to English for short texts
        
    # Count character frequencies
    en_chars = len(re.findall(r'[a-zA-Z]', text))
    # Detect languages with specific scripts
    if re.search(r'[\u0900-\u097F]', text):  # Devanagari (Hindi)
        return "hi"
    elif re.search(r'[\u0600-\u06FF]', text):  # Arabic script
        return "ar"
    elif re.search(r'[\u0400-\u04FF]', text):  # Cyrillic (Russian)
        return "ru"
    elif re.search(r'[\u4E00-\u9FFF]', text):  # Chinese
        return "zh"
    elif re.search(r'[\u3040-\u30FF]', text):  # Japanese
        return "ja"
    elif re.search(r'[\uAC00-\uD7AF]', text):  # Korean
        return "ko"
    elif re.search(r'[\u0E00-\u0E7F]', text):  # Thai
        return "th"
    
    # Default to English
    return "en"

def extract_text_from_page(page, exclude_headers_footers=True):
    """Extract clean text from a page, optionally excluding headers/footers"""
    text_blocks = []
    
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)["blocks"]
    
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                line_text = ""
                y_pos = line["bbox"][1]
                
                # Skip headers/footers
                if exclude_headers_footers and (y_pos < 50 or y_pos > page.rect.height - 50):
                    continue
                
                for span in line["spans"]:
                    line_text += span["text"]
                
                if line_text.strip():
                    text_blocks.append(line_text.strip())
    
    return " ".join(text_blocks)

def detect_toc_page(doc):
    """Detect pages that likely contain a table of contents"""
    toc_pages = []
    
    # Regular expressions to identify TOC patterns
    toc_patterns = [
        re.compile(r'\bcontents\b', re.IGNORECASE),
        re.compile(r'\btable\s+of\s+contents\b', re.IGNORECASE),
        re.compile(r'\bindex\b', re.IGNORECASE),
        re.compile(r'^\s*\d+(\.\d+)*\s+[A-Za-z].*\.{2,}\s*\d+\s*$'),  # "1.2 Title......42" pattern
    ]
    
    for page_num in range(min(10, len(doc))):  # Check first 10 pages
        page = doc.load_page(page_num)
        text = page.get_text("text")
        
        # Check for TOC headers
        if any(pattern.search(text) for pattern in toc_patterns[:3]):
            toc_pages.append(page_num)
            continue
            
        # Check for dot leader patterns (common in TOCs)
        dot_leader_lines = 0
        for line in text.split('\n'):
            if re.search(r'\.{3,}\s*\d+$', line):  # "Title.........42" pattern
                dot_leader_lines += 1
        
        if dot_leader_lines >= 3:  # Multiple dot leader lines suggest a TOC
            toc_pages.append(page_num)
    
    return toc_pages

def extract_pdf_structure(pdf_path):
    doc = fitz.open(pdf_path)
    title = ""
    headings = []
    toc_headings = []  # Store headings from TOC separately
    font_stats = defaultdict(int)
    text_blocks_by_page = []
    
    # Get document metadata and page count for later use
    doc_info = doc.metadata
    page_count = len(doc)
    logger.info(f"Processing document with {page_count} pages")
    
    # Check if the document has a built-in table of contents
    built_in_toc = doc.get_toc()
    if built_in_toc:
        logger.info(f"Document has a built-in table of contents with {len(built_in_toc)} entries")
        
        # Extract headings from built-in TOC
        for level, title_text, page_num in built_in_toc:
            # Convert level to H1, H2, H3 format
            if level == 1:
                heading_level = "H1"
            elif level == 2:
                heading_level = "H2"
            else:
                heading_level = "H3"
                
            # Store TOC heading
            toc_headings.append({
                "level": heading_level,
                "text": title_text,
                "page": page_num,
                "from_toc": True,
                "confidence": "high"  # Built-in TOC has high confidence
            })
        
        # Set title from metadata or first H1
        if doc_info and 'title' in doc_info and doc_info['title']:
            title = doc_info['title']
        elif toc_headings and toc_headings[0]["level"] == "H1":
            title = toc_headings[0]["text"]
    
    # Detect TOC pages to potentially skip them in analysis
    toc_pages = detect_toc_page(doc)
    if toc_pages:
        logger.info(f"Detected possible Table of Contents on pages: {[p+1 for p in toc_pages]}")
    
    # First pass: Collect font statistics and all text blocks
    for page_num in range(len(doc)):
        # Skip TOC pages for font statistics
        if page_num in toc_pages:
            text_blocks_by_page.append([])
            continue
            
        page = doc.load_page(page_num)
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)["blocks"]
        page_blocks = []
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_stats[span["size"]] += 1
                        
                    line_text = "".join(span["text"] for span in line["spans"])
                    if line_text.strip():
                        max_size = max([span["size"] for span in line["spans"]], default=0)
                        is_bold = any(span["flags"] & 16 for span in line["spans"])  # Check bold flag
                        is_italic = any(span["flags"] & 1 for span in line["spans"])  # Check italic flag
                        
                        page_blocks.append({
                            "text": line_text.strip(),
                            "size": max_size,
                            "is_bold": is_bold,
                            "is_italic": is_italic,
                            "y0": line["bbox"][1],
                            "y1": line["bbox"][3],
                            "x0": line["bbox"][0],  # Add x-coordinates for indent detection
                            "x1": line["bbox"][2],
                            "spans": line["spans"],
                            "font": line["spans"][0]["font"] if line["spans"] else ""
                        })
        
        text_blocks_by_page.append(page_blocks)
    
    # Analyze layout patterns
    visual_gaps = calculate_visual_gap_features(text_blocks_by_page)
    geometric_layout = analyze_geometric_layout(text_blocks_by_page)
    font_analysis = analyze_font_consistency([block for page in text_blocks_by_page for block in page])
    
    # Try to extract outline directly from TOC page if available
    toc_page_headings = []
    if toc_pages:
        toc_page_headings = detect_outline_from_toc_page(doc, text_blocks_by_page)
        if toc_page_headings:
            logger.info(f"Extracted {len(toc_page_headings)} headings from TOC page")
    
    # Analyze font sizes and determine thresholds more precisely
    if font_stats:
        # Sort sizes by frequency to find the most common (body text)
        size_freq = sorted([(size, count) for size, count in font_stats.items()], 
                          key=lambda x: x[1], reverse=True)
        
        # Get most common font size (likely body text)
        body_size = size_freq[0][0]
        
        # Sort sizes by value (descending)
        sorted_sizes = sorted(font_stats.keys(), reverse=True)
        
        # Create clusters of similar font sizes to account for minor variations
        clusters = []
        current_cluster = [sorted_sizes[0]]
        
        for i in range(1, len(sorted_sizes)):
            if abs(sorted_sizes[i] - current_cluster[-1]) < 0.5:  # Within 0.5 pt = same size
                current_cluster.append(sorted_sizes[i])
            else:
                # Calculate cluster average and count
                avg_size = sum(current_cluster) / len(current_cluster)
                count = sum(font_stats[s] for s in current_cluster)
                clusters.append((avg_size, count))
                current_cluster = [sorted_sizes[i]]
        
        # Add the last cluster
        if current_cluster:
            avg_size = sum(current_cluster) / len(current_cluster)
            count = sum(font_stats[s] for s in current_cluster)
            clusters.append((avg_size, count))
            
        # Sort clusters by size
        clusters.sort(reverse=True)
        
        # Determine heading thresholds based on size clusters
        # Different strategies depending on number of clusters found
        if len(clusters) >= 4:  # Many distinct sizes
            h1_threshold = clusters[0][0] * 0.9  # Largest size
            h2_threshold = clusters[1][0] * 0.9  # Second largest
            h3_threshold = clusters[2][0] * 0.9  # Third largest
        elif len(clusters) == 3:
            h1_threshold = clusters[0][0] * 0.9
            h2_threshold = clusters[1][0] * 0.9
            h3_threshold = (clusters[1][0] + clusters[2][0]) / 2  # Between 2nd and 3rd
        elif len(clusters) == 2:
            h1_threshold = clusters[0][0] * 0.9
            h2_threshold = (clusters[0][0] + clusters[1][0]) / 2
            h3_threshold = clusters[1][0] * 1.1  # Slightly larger than body text
        else:
            # Only one cluster - use proportions from body text
            h1_threshold = body_size * 1.5
            h2_threshold = body_size * 1.3
            h3_threshold = body_size * 1.1
            
        # Check if body text is actually in the clusters
        body_size_cluster = max(clusters, key=lambda x: x[1])[0]
        
        logger.info(f"Font analysis - Body: {body_size_cluster:.2f}, H1: {h1_threshold:.2f}, "
                   f"H2: {h2_threshold:.2f}, H3: {h3_threshold:.2f}")
    else:
        h1_threshold, h2_threshold, h3_threshold = 20, 16, 14
        body_size = 12
    
    # Regular expressions for heading detection
    h1_patterns = [
        re.compile(r'^chapter\s+\d+\b', re.IGNORECASE),
        re.compile(r'^section\s+\d+\b', re.IGNORECASE),
        re.compile(r'^part\s+\d+:', re.IGNORECASE),
        re.compile(r'^\d+\.\s+[A-Z]'),  # e.g. "1. INTRODUCTION"
    ]
    
    h2_patterns = [
        re.compile(r'^\d+\.\d+\s+[A-Z]'),  # e.g. "1.1 SUBSECTION"
        re.compile(r'^[A-Z][a-z]+\s+\d+\.\d+:'),  # e.g. "Section 1.2:"
        re.compile(r'^[A-Za-z]+\s+\d+:'),  # e.g. "Table 1:" or "Figure 2:"
    ]
    
    h3_patterns = [
        re.compile(r'^\d+\.\d+\.\d+\s+\w+'),  # e.g. "1.1.1 Subsubsection"
        re.compile(r'^•\s+[A-Z][a-z]+:'),  # e.g. "• Note:"
        re.compile(r'^\[\w+\]:'),  # e.g. "[NOTE]:"
    ]
    
    # Keep track of hierarchy for context
    current_h1 = None
    current_h2 = None
    heading_paths = {}  # To track heading paths for hierarchy
    
    # Calculate average indentation for each page
    page_indents = []
    for page_blocks in text_blocks_by_page:
        if not page_blocks:
            page_indents.append((0, 0, 0))  # Default
            continue
            
        indents = [block["x0"] for block in page_blocks]
        if indents:
            min_indent = min(indents)
            avg_indent = sum(indents) / len(indents)
            max_indent = max(indents)
            page_indents.append((min_indent, avg_indent, max_indent))
        else:
            page_indents.append((0, 0, 0))
    
    # Second pass: Identify headings using multiple analysis techniques
    for page_num, page_blocks in enumerate(text_blocks_by_page):
        min_indent, avg_indent, max_indent = page_indents[page_num]
        
        # Skip empty pages or TOC pages
        if not page_blocks or page_num in toc_pages:
            continue
        
        # Get visual gap features for this page
        page_gaps = visual_gaps.get(page_num, {"significant_gaps": []})
        significant_gaps = page_gaps.get("significant_gaps", [])
        
        # Get geometric layout for this page
        page_layout = geometric_layout.get(page_num, {})
        indent_levels = page_layout.get("indent_levels", {})
        
        for i, block in enumerate(page_blocks):
            # Skip headers/footers
            if block["y0"] < 50 or block["y0"] > doc[page_num].rect.height - 50:
                continue
                
            text = block["text"]
            clean_text = re.sub(r'\s+', ' ', text).strip()
            
            # Skip very long text (likely paragraph) or empty text
            if not clean_text or len(clean_text) > 200:
                continue
                
            # Skip page numbers and standalone dates
            if re.match(r'^\d+$', clean_text) or re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', clean_text):
                continue
            
            # Check for section heading numbering patterns (1.2.3)
            section_match = re.match(r'^(\d+)(?:\.(\d+)(?:\.(\d+))?)?', clean_text)
            section_level = 0
            if section_match:
                # Count how many groups matched to determine section depth
                section_level = sum(1 for g in section_match.groups() if g is not None)
            
            # Get linguistic features
            linguistic_analysis = analyze_text_linguistic_features(clean_text)
            
            # Determine heading level using multiple signals
            level = None
            confidence = "low"
            
            # Check if block is near a significant visual gap
            is_after_gap = any(j == i for _, j in significant_gaps)
            
            # First check font size
            if block["size"] >= h1_threshold:
                level = "H1"
                confidence = "medium"
            elif block["size"] >= h2_threshold:
                level = "H2"
                confidence = "medium"
            elif block["size"] >= h3_threshold:
                level = "H3"
                confidence = "low"
            
            # Check font category from consistency analysis
            font_key = (block["size"], block.get("is_bold", False), block.get("is_italic", False))
            if "font_categories" in font_analysis and font_key in font_analysis["font_categories"]:
                font_category = font_analysis["font_categories"][font_key]
                if font_category == "h1" and level != "H1":
                    level = "H1"
                    confidence = max(confidence, "medium")
                elif font_category == "h2" and (level is None or level == "H3"):
                    level = "H2"
                    confidence = max(confidence, "medium")
                elif font_category == "h3" and level is None:
                    level = "H3"
                    confidence = max(confidence, "medium")
            
            # Check patterns for specific heading levels
            if any(pattern.match(clean_text) for pattern in h1_patterns):
                level = level or "H1"
                confidence = "high"
            elif any(pattern.match(clean_text) for pattern in h2_patterns):
                level = level or "H2"
                confidence = "high"
            elif any(pattern.match(clean_text) for pattern in h3_patterns):
                level = level or "H3"
                confidence = "high"
                
            # Apply section numbering logic
            if section_level > 0:
                if section_level == 1:
                    level = level or "H1"
                    confidence = max(confidence, "high")
                elif section_level == 2:
                    level = level or "H2"
                    confidence = max(confidence, "high")
                elif section_level >= 3:
                    level = level or "H3"
                    confidence = max(confidence, "high")
            
            # Check for bullet points at the beginning (often H3)
            if re.match(r'^[•\-*]\s', clean_text):
                if not level or level == "H3":  # Only upgrade to H3, not downgrade from H1/H2
                    level = "H3"
                    confidence = max(confidence, "medium")
            
            # Use linguistic analysis to confirm or adjust level
            if linguistic_analysis["likely_heading"]:
                if level is None:
                    # If high confidence it's a heading but level not set
                    if linguistic_analysis["confidence"] > 0.4:
                        level = "H3"  # Default to H3 if unsure
                        confidence = max(confidence, "medium")
                else:
                    # Increase confidence based on linguistic analysis
                    if linguistic_analysis["confidence"] > 0.6:
                        confidence = max(confidence, "high")
                    elif linguistic_analysis["confidence"] > 0.3:
                        confidence = max(confidence, "medium")
            else:
                # If linguistic analysis strongly suggests not a heading
                if linguistic_analysis["confidence"] > 0.5 and confidence != "high":
                    level = None
            
            # Check for short, bold text (potential headings)
            if block["is_bold"] and len(clean_text) < 80:
                # Check isolation (space before/after)
                is_isolated = False
                
                if i == 0 or i == len(page_blocks) - 1:
                    is_isolated = True
                elif i > 0 and i < len(page_blocks) - 1:
                    prev_block = page_blocks[i-1]
                    next_block = page_blocks[i+1]
                    if block["y0"] - prev_block["y1"] > 10 and next_block["y0"] - block["y1"] > 10:
                        is_isolated = True
                
                if is_isolated:
                    if not level:  # Only set if not determined by other rules
                        # More likely to be H2 than H3 if isolated and bold
                        level = "H2"
                        confidence = max(confidence, "medium")
                    confidence = max(confidence, "medium")  # Increase confidence if bold and isolated
            
            # Special case for capitalized short text (often headings)
            if clean_text.isupper() and 3 < len(clean_text) < 50:
                if not level:  # Only set if not determined by other rules
                    level = "H2"
                    confidence = max(confidence, "medium")
            
            # Check indentation (more indented often means lower level heading)
            indent_ratio = 0
            if max_indent > min_indent:
                indent_ratio = (block["x0"] - min_indent) / (max_indent - min_indent)
                
                # Use indentation to help determine level if not already determined
                if indent_ratio > 0.6 and not level:
                    level = "H3"  # Deeply indented
                elif indent_ratio > 0.3 and not level:
                    level = "H2"  # Moderately indented
            
            # Add visual gap information to adjust confidence
            if is_after_gap and level:
                confidence = max(confidence, "medium")
                
            # Check if this follows our hierarchy expectation
            if level:
                # Look at text following potential heading
                following_text = ""
                following_size = 0
                if i < len(page_blocks) - 1:
                    next_block = page_blocks[i+1]
                    following_text = next_block["text"]
                    following_size = next_block["size"]
                
                # If followed by larger text, probably not a heading
                if following_size > block["size"] * 1.2:
                    confidence = min(confidence, "low")
                
                # If short text followed by substantial normal-sized text, more likely a heading
                if len(following_text) > 50 and abs(following_size - body_size) < 1.0:
                    confidence = max(confidence, "medium")
                
                # Calculate the actual indent level if available
                indent_level = 0
                if len(block["spans"]) > 0 and "origin" in block["spans"][0]:
                    x_pos = block["spans"][0]["origin"][0]
                    # Check if this is indented compared to other text
                    indent_level = round(x_pos / 10)  # Rough approximation
                
                # Use indent to adjust heading level for borderline cases
                if confidence == "low" and indent_level > 2:
                    # Deeply indented text is more likely to be H3 than H2, H1
                    if level == "H2":
                        level = "H3"
                    
                # Track heading hierarchy for context
                if level == "H1":
                    current_h1 = clean_text
                    current_h2 = None
                elif level == "H2":
                    current_h2 = clean_text
                
                # Create heading path for context
                path = []
                if current_h1:
                    path.append(current_h1)
                if current_h2 and level == "H3":
                    path.append(current_h2)
                
                heading_path = " > ".join(path)
                
                # Only include headings with sufficient confidence
                if confidence != "low":
                    headings.append({
                        "level": level,
                        "text": clean_text,
                        "page": page_num + 1,
                        "confidence": confidence,
                        "path": heading_path if path else "",
                        "y0": block["y0"],
                        "from_toc": False
                    })
                    
                    # Set title from first H1 or biggest text on first page
                    if not title and (level == "H1" or (page_num == 0 and i == 0)):
                        title = clean_text
    
    # Combine all heading sources with priority
    combined_headings = []
    
    # First, add headings from built-in TOC if available (highest priority)
    if toc_headings:
        combined_headings.extend(toc_headings)
    
    # Next, add headings from TOC page extraction if available
    if toc_page_headings:
        # Filter out duplicates from built-in TOC
        toc_texts = {h["text"].lower(): (h["level"], h["page"]) for h in toc_headings}
        for h in toc_page_headings:
            h_key = h["text"].lower()
            is_duplicate = any(h_key == t_key or (h_key in t_key or t_key in h_key) 
                           for t_key in toc_texts.keys())
            if not is_duplicate:
                combined_headings.append(h)
    
    # Add visual headings that don't duplicate TOC headings
    existing_texts = {h["text"].lower(): (h["level"], h["page"]) for h in combined_headings}
    
    for h in headings:
        # Check if this heading is already in combined headings
        h_key = h["text"].lower()
        is_duplicate = False
        
        for existing_key, (level, page) in existing_texts.items():
            # Check for exact match or if heading contains/is contained in existing entry
            if ((h_key == existing_key or 
                (h_key in existing_key or existing_key in h_key)) and h["page"] == page):
                is_duplicate = True
                break
        
        if not is_duplicate:
            combined_headings.append(h)
    
    # If we didn't get anything useful, use just the visual headings
    if not combined_headings:
        combined_headings = headings
    
    # Detect document language
    document_language = "en"  # Default
    if page_count > 0:
        try:
            first_page = doc[0]
            first_page_text = extract_text_from_page(first_page)
            document_language = detect_language(first_page_text)
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
    
    # Check heading sequence patterns and fix inconsistencies
    heading_suggestions = identify_heading_sequences(combined_headings)
    if heading_suggestions:
        logger.info(f"Found {len(heading_suggestions)} heading sequence issues")
        
        # Apply automatic corrections where possible
        for suggestion in heading_suggestions:
            if suggestion["issue"] == "missing_parent":
                # Create missing parent headings
                for missing_h1 in suggestion["missing"]:
                    # Find first H2 with this missing parent
                    for h in combined_headings:
                        if h["level"] == "H2" and h["text"].startswith(f"{missing_h1}."):
                            # Extract the H1 title
                            match = re.match(r'^(\d+)\.', h["text"])
                            if match:
                                h1_text = f"{match.group(1)}. Section"  # Generic title
                                combined_headings.append({
                                    "level": "H1",
                                    "text": h1_text,
                                    "page": h["page"],
                                    "confidence": "medium",
                                    "from_toc": False,
                                    "inferred": True
                                })
                                break
    
    # Post-process headings to ensure proper hierarchy and remove duplicates
    filtered_headings = []
    seen_texts = set()
    
    # First, sort headings by page and position to ensure correct order
    combined_headings.sort(key=lambda h: (h["page"], h.get("y0", 0)))
    
    # Remove duplicates and very similar headings
    for heading in combined_headings:
        # Create a unique key based on heading text and page
        heading_key = f"{heading['text'].lower()[:50]}_{heading['page']}"
        
        # Check for very similar text on same page (e.g. "1. Introduction" vs "Introduction")
        duplicate = False
        for existing_key in seen_texts:
            existing_text, existing_page = existing_key.rsplit('_', 1)
            current_text = heading['text'].lower()[:50]
            current_page = str(heading['page'])
            
            # If on same page and one text contains the other
            if existing_page == current_page and (
                existing_text in current_text or current_text in existing_text):
                duplicate = True
                break
                
        if not duplicate and heading_key not in seen_texts:
            seen_texts.add(heading_key)
            # Remove temporary fields
            for field in ["confidence", "path", "y0", "from_toc", "inferred"]:
                if field in heading:
                    del heading[field]
            filtered_headings.append(heading)
    
    # Ensure heading hierarchy makes sense - never have H3 directly after H1
    for i in range(1, len(filtered_headings)):
        prev_heading = filtered_headings[i-1]
        curr_heading = filtered_headings[i]
        
        # If H1 followed by H3, change H3 to H2
        if prev_heading["level"] == "H1" and curr_heading["level"] == "H3":
            filtered_headings[i]["level"] = "H2"
            
    # If no H1 found but H2 exists, promote first H2 to H1
    has_h1 = any(h["level"] == "H1" for h in filtered_headings)
    if not has_h1 and filtered_headings:
        # Find first H2
        for i, heading in enumerate(filtered_headings):
            if heading["level"] == "H2":
                filtered_headings[i]["level"] = "H1"
                break
    
    # Check document metadata for title
    if doc_info and 'title' in doc_info and doc_info['title'] and doc_info['title'].strip():
        title = doc_info['title']
    # Fallback for title if not found
    elif not title and filtered_headings:
        title = filtered_headings[0]["text"]
    elif not title:
        title = os.path.basename(pdf_path).replace('.pdf', '')
    
    # Create the result dictionary
    result = {
        "title": title,
        "outline": filtered_headings,
        "metadata": {
            "language": document_language,
            "pages": page_count,
            "has_built_in_toc": len(built_in_toc) > 0 if built_in_toc else False
        }
    }
    
    # Close the document
    doc.close()
    
    return result

def process_pdfs():
    # Detect environment (Docker vs local)
    if os.path.exists('/app/input'):
        input_dir = '/app/input'
        output_dir = '/app/output'
    else:
        input_dir = 'input'
        output_dir = 'output'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            try:
                result = extract_pdf_structure(pdf_path)
                
                # Post-process: Remove likely false positive headings
                # Filter out headings that appear to be part of paragraphs
                filtered_result = filter_false_positives(result)
                
                # For Docker compatibility - only include required fields in output
                final_result = {
                    "title": filtered_result["title"],
                    "outline": filtered_result["outline"]
                }
                
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(final_result, f, indent=2, ensure_ascii=False)
                
                processed_count += 1
                logger.info(f"Processed: {filename} - Found {len(filtered_result['outline'])} headings")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
    
    logger.info(f"Successfully processed {processed_count} PDF files")

def filter_false_positives(result):
    """Filter out likely false positive headings that are just part of paragraphs"""
    if not result or "outline" not in result:
        return result
        
    outline = result["outline"]
    
    # 1. Group headings by page to identify context
    headings_by_page = defaultdict(list)
    for h in outline:
        headings_by_page[h["page"]].append(h)
    
    # 2. Sort each page's headings
    for page, page_headings in headings_by_page.items():
        page_headings.sort(key=lambda h: h.get("y0", 0))
    
    # 3. Identify patterns that suggest false positives
    filtered_outline = []
    skip_indices = set()
    
    # First, identify headings to skip
    for page, page_headings in headings_by_page.items():
        # Skip pages with just 1-2 headings (likely real headings)
        if len(page_headings) <= 2:
            continue
            
        # Check for paragraph-like sequences (many H2/H3 in sequence)
        h2_h3_sequence_count = 0
        prev_h_idx = None
        
        for i, h in enumerate(page_headings):
            if h["level"] in ["H2", "H3"]:
                h2_h3_sequence_count += 1
                
                # Check for sentence continuation patterns
                text = h["text"].lower()
                
                # More aggressive paragraph pattern detection
                is_likely_paragraph = False
                
                # Likely paragraph indicators (expanded)
                common_sentence_starters = ('the', 'a ', 'an ', 'in ', 'on ', 'with ', 'and ', 'or ', 'for ', 
                                           'to ', 'that ', 'this ', 'these ', 'those ', 'it ', 'they ', 'we ', 
                                           'you ', 'he ', 'she ', 'as ', 'by ', 'from ')
                
                # Check for more paragraph-like patterns
                if (
                    # Starts with lowercase conjunction/preposition/article
                    (not text.startswith(('•', '-', '*', '1.', '2.', '3.', 'chapter', 'section')) and
                    (text.startswith(common_sentence_starters) or
                     # Long text without strong heading indicators
                     (len(text.split()) > 8 and not text.endswith(':') and
                      not re.match(r'^\d+(\.\d+)*\s', text)) or  
                     # Doesn't start with capital and not a bullet
                     (not text[0].isupper() and not text.startswith(('•', '-', '*'))))) or
                    # Contains typical sentence fragments
                    (' of the ' in text or ' in the ' in text or ' to be ' in text or
                    ' as well as ' in text or ' such as ' in text or ' in order to ' in text) or
                    # Ends with connector like 'and', 'or', 'but'
                    text.endswith((' and', ' or', ' but', ' nor', ' yet', ' so'))
                ):
                    is_likely_paragraph = True
                
                if is_likely_paragraph:
                    # Check if previous heading exists
                    if prev_h_idx is not None:
                        prev_h = page_headings[prev_h_idx]
                        
                        # If previous heading ends without punctuation and this one doesn't start with a capital
                        # or previous heading doesn't end with punctuation
                        prev_text = prev_h["text"]
                        if (not prev_text.endswith(('.', '!', '?', ':', ';')) or 
                            (not text[0].isupper() and not text.startswith(('•', '-', '*')))):
                            
                            # Mark both as likely paragraph fragments
                            skip_indices.add((page, i))
                            skip_indices.add((page, prev_h_idx))
                
                prev_h_idx = i
            else:
                # Reset sequence count for H1
                h2_h3_sequence_count = 0
                prev_h_idx = i
                
        # If there are too many H2/H3 on a page (relative to page count), they're likely paragraphs
        if h2_h3_sequence_count > 10 and len(page_headings) > 15:
            # Keep only the most likely headings (those not in skip_indices)
            for i, h in enumerate(page_headings):
                if h["level"] in ["H2", "H3"] and (page, i) not in skip_indices:
                    # Additional checks for likely headings
                    text = h["text"]
                    
                    # Strong heading indicators
                    strong_heading_indicators = (
                        text.endswith(':') or  # Ends with colon
                        re.match(r'^\d+\.\d+', text) or  # Numbered like "1.2"
                        re.match(r'^[A-Z][a-z]+:', text) or  # "Title:" format
                        re.match(r'^[•\-*]\s+[A-Z]', text) or  # Bullet with capital
                        text.isupper() or  # ALL CAPS
                        len(text.split()) <= 4 or  # Very short
                        re.match(r'^(chapter|section|part|appendix|figure|table)\s+\d+', text.lower())  # Standard heading prefix
                    )
                    
                    if not strong_heading_indicators:
                        skip_indices.add((page, i))
    
    # Special case: if all H2/H3 on a page are marked for skipping, keep the first one
    # (likely the main section heading followed by paragraph text)
    for page, page_headings in headings_by_page.items():
        h2h3_headings = [(i, h) for i, h in enumerate(page_headings) if h["level"] in ["H2", "H3"]]
        if h2h3_headings and all((page, i) in skip_indices for i, _ in h2h3_headings):
            # Keep the first one
            skip_indices.remove((page, h2h3_headings[0][0]))
    
    # Create the filtered outline
    for page, page_headings in headings_by_page.items():
        for i, h in enumerate(page_headings):
            if (page, i) not in skip_indices:
                filtered_outline.append(h)
    
    # Ensure proper hierarchy in final output
    if filtered_outline:
        # Sort by page
        filtered_outline.sort(key=lambda h: h["page"])
        
        # Check if we have an H1
        has_h1 = any(h["level"] == "H1" for h in filtered_outline)
        if not has_h1 and filtered_outline:
            # Promote first heading to H1
            filtered_outline[0]["level"] = "H1"
    
    # Add metadata about filtering
    filtered_pct = 0
    if outline:
        filtered_pct = round((len(outline) - len(filtered_outline)) / len(outline) * 100)
        logger.info(f"Filtered {filtered_pct}% of detected headings as false positives")
    
    result["outline"] = filtered_outline
    return result

if __name__ == "__main__":
    process_pdfs()