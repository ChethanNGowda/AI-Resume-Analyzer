# ATS (Applicant Tracking System) Optimization Guide

## How ATS Systems Process Resumes
1. **Text Extraction & Parsing**: ATS parsers (Workday, Taleo, Greenhouse, Lever, Ashby) convert PDF or DOCX files into raw text strings and parse them into standard schema fields (Work Experience, Skills, Education).
2. **Keyword & Semantic Matching**: The ATS calculates relevance scores based on presence, frequency, and contextual placement of skills required in the job description.

## Critical Formatting Guidelines
- **Single-Column Layout**: Multi-column layouts often cause ATS parsers to read across columns horizontally, scrambling text.
- **Avoid Tables, Text Boxes, Headers/Footers**: Many parsers ignore content placed inside header/footer bands or graphical text frames.
- **Standard Headings**: Use predictable headings: "Experience", "Skills", "Education", "Projects", "Certifications".
- **ATS-Friendly File Formats**: Standard text-based PDF or clean .docx. Avoid scanned image PDFs without selectable text.
- **Bullet Character Safety**: Use standard circular or dash bullets rather than custom wingdings, checkmarks, or icons.
- **Target Keyword Mirroring**: If the job specifies "PostgreSQL" and "Docker", write "PostgreSQL" and "Docker" verbatim rather than just "SQL" and "containers".
