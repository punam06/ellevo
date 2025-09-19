import pdfplumber

def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def find_tasks(text):
    """Find tasks 1-5 in the extracted text"""
    lines = text.split('\n')
    tasks = []
    current_task = None
    current_task_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for task headers (Task 1, Task 2, etc.)
        if line.lower().startswith('task '):
            # Save previous task if exists
            if current_task and current_task_content:
                tasks.append({
                    'number': current_task,
                    'content': '\n'.join(current_task_content)
                })
            
            # Start new task
            current_task = line
            current_task_content = [line]
        elif current_task and line:
            current_task_content.append(line)
    
    # Save the last task
    if current_task and current_task_content:
        tasks.append({
            'number': current_task,
            'content': '\n'.join(current_task_content)
        })
    
    return tasks

if __name__ == "__main__":
    pdf_path = "Natural Language Processing Tasks.pdf"
    
    print("Extracting text from PDF...")
    text = extract_pdf_text(pdf_path)
    
    print("Finding tasks 1-5...")
    tasks = find_tasks(text)
    
    print(f"Found {len(tasks)} tasks:")
    print("="*50)
    
    for i, task in enumerate(tasks[:5]):  # Only show first 5 tasks
        print(f"\n{task['number']}")
        print("-" * len(task['number']))
        print(task['content'])
        print("="*50)
    
    # Save full text to file for reference
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    print("\nFull extracted text saved to 'extracted_text.txt'")