# src/resume_parser.py
import pdfplumber
import re

SKILLS_VOCAB = {
    'python','java','sql','javascript','react','docker','kubernetes','aws',
    'tensorflow','pytorch','pandas','numpy','spark','spring','kotlin','android',
    'html','css','django','flask','nodejs','express','terraform','ci/cd',
    'selenium','manual testing','rest','oop','dsa','data structures and algorithms'
}

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def find_skills(text):
    text_lower = text.lower()
    found = set()
    for s in SKILLS_VOCAB:
        if s in text_lower:
            found.add(s)
    m = re.search(r"(skills|technical skills|technologies)[:\-]\s*([^\n\r]+)", text_lower)
    if m:
        toks = re.split(r"[,\|;/]", m.group(2))
        for t in toks:
            tok = t.strip().replace('.', '')
            if tok in SKILLS_VOCAB:
                found.add(tok)
            parts = tok.split()
            for p in parts:
                if p in SKILLS_VOCAB:
                    found.add(p)
    return found

def extract_profile_from_pdf(path):
    txt = extract_text_from_pdf(path)
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    title = lines[0] if lines else ""
    bio = " ".join(lines[1:4])[:500]
    skills = find_skills(txt)
    return {"title": title, "bio": bio, "skills": ", ".join(sorted(skills))}
