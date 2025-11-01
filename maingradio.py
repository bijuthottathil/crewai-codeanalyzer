"""
AI Code Review System - Standalone Gradio Version
Just Gradio, no FastAPI complications!
"""

import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List
import gradio as gr
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def clone_github_repo(repo_url: str, target_dir: str) -> bool:
    """Clone a GitHub repository."""
    try:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        subprocess.run(
            ["git", "clone", repo_url, target_dir],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except Exception as e:
        print(f"Error cloning repository: {str(e)}")
        return False


def get_repository_structure(repo_path: str) -> str:
    """Generate repository tree structure."""
    structure = []
    repo_path_obj = Path(repo_path)
    
    def build_tree(directory: Path, prefix: str = "", is_last: bool = True):
        if directory.name.startswith('.'):
            return
        
        connector = "└── " if is_last else "├── "
        structure.append(f"{prefix}{connector}{directory.name}")
        
        if directory.is_dir():
            try:
                children = sorted(list(directory.iterdir()), 
                                key=lambda x: (not x.is_dir(), x.name))
                children = [c for c in children if not c.name.startswith('.')]
                
                for i, child in enumerate(children):
                    extension = "    " if is_last else "│   "
                    build_tree(child, prefix + extension, i == len(children) - 1)
            except PermissionError:
                pass
    
    structure.append(repo_path_obj.name)
    try:
        children = sorted(list(repo_path_obj.iterdir()), 
                        key=lambda x: (not x.is_dir(), x.name))
        children = [c for c in children if not c.name.startswith('.')]
        
        for i, child in enumerate(children):
            build_tree(child, "", i == len(children) - 1)
    except Exception:
        pass
    
    return "\n".join(structure)


def read_code_files(repo_path: str) -> Dict[str, str]:
    """Read all code files from repository."""
    extensions = ['.py', '.js', '.java', '.cpp', '.c', '.go', '.rs', '.ts', 
                 '.jsx', '.tsx', '.php', '.rb', '.swift', '.kt']
    
    code_files = {}
    repo_path_obj = Path(repo_path)
    
    for file_path in repo_path_obj.rglob('*'):
        if file_path.is_file() and file_path.suffix in extensions:
            if any(part.startswith('.') or part in ['node_modules', '__pycache__', 'venv', 'env'] 
                   for part in file_path.parts):
                continue
            
            try:
                relative_path = file_path.relative_to(repo_path_obj)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if len(content) > 50000:
                        content = content[:50000] + "\n... [truncated]"
                    code_files[str(relative_path)] = content
            except Exception:
                pass
    
    return code_files


def create_code_summary(code_files: Dict[str, str], max_files: int = 20) -> str:
    """Create detailed code summary."""
    priority_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java']
    priority_names = ['main', 'app', 'server', 'index', 'config']
    
    def get_priority(file_path: str) -> int:
        score = 0
        path_lower = file_path.lower()
        
        for ext in priority_extensions:
            if path_lower.endswith(ext):
                score += 10
                break
        
        for name in priority_names:
            if name in path_lower:
                score += 5
                break
        
        return score
    
    sorted_files = sorted(code_files.items(), key=lambda x: get_priority(x[0]), reverse=True)
    selected_files = sorted_files[:max_files]
    
    summary_parts = []
    for file_path, content in selected_files:
        display_content = content[:3000]
        if len(content) > 3000:
            display_content += "\n... [truncated]"
        
        summary_parts.append(f"""
{'='*70}
FILE: {file_path}
{'='*70}
{display_content}
""")
    
    return "\n".join(summary_parts)


def get_code_stats(code_files: Dict[str, str]) -> str:
    """Generate code statistics."""
    total_lines = 0
    total_files = len(code_files)
    files_by_type = {}
    
    for file_path, content in code_files.items():
        lines = len(content.split('\n'))
        total_lines += lines
        
        ext = Path(file_path).suffix
        if ext not in files_by_type:
            files_by_type[ext] = {'count': 0, 'lines': 0}
        files_by_type[ext]['count'] += 1
        files_by_type[ext]['lines'] += lines
    
    stats = f"""
CODE STATISTICS:
- Total Files: {total_files}
- Total Lines: {total_lines:,}
- Avg Lines/File: {total_lines // total_files if total_files > 0 else 0}

FILES BY TYPE:
"""
    for ext, data in sorted(files_by_type.items(), key=lambda x: x[1]['lines'], reverse=True):
        stats += f"  {ext}: {data['count']} files, {data['lines']:,} lines\n"
    
    return stats


def analyze_repository(repo_url: str, progress=gr.Progress()) -> str:
    """Main analysis function with progress tracking."""
    
    temp_dir = tempfile.mkdtemp()
    repo_path = os.path.join(temp_dir, "repo")
    
    try:
        # Step 1: Clone
        progress(0.1, desc="🔄 Cloning repository...")
        
        if not clone_github_repo(repo_url, repo_path):
            return "❌ Failed to clone repository. Check URL and ensure it's public."
        
        # Step 2: Scan structure
        progress(0.2, desc="📁 Scanning repository structure...")
        repo_structure = get_repository_structure(repo_path)
        
        # Step 3: Read files
        progress(0.3, desc="📄 Reading code files...")
        code_files = read_code_files(repo_path)
        
        if not code_files:
            return "⚠️ No code files found in repository."
        
        # Step 4: Prepare analysis
        progress(0.4, desc=f"📊 Preparing analysis ({len(code_files)} files)...")
        code_summary = create_code_summary(code_files, max_files=20)
        code_stats = get_code_stats(code_files)
        
        # Step 5: Initialize AI
        progress(0.5, desc="🤖 Initializing AI agents (gpt-4.1-nano)...")
        
        llm = ChatOpenAI(
            model="gpt-4.1-nano",
            temperature=0.1,
            max_tokens=4000
        )
        
        # Define agents
        arch_agent = Agent(
            role="Senior Software Architect",
            goal="Find specific architecture issues with file locations and code examples",
            backstory="You provide SPECIFIC recommendations with exact file:line locations and code snippets.",
            llm=llm,
            verbose=True,
        )
        
        quality_agent = Agent(
            role="Code Quality Expert",
            goal="Find specific code quality issues with exact locations",
            backstory="You identify specific functions, variables, and code blocks that need improvement.",
            llm=llm,
            verbose=True,
        )
        
        security_agent = Agent(
            role="Security Specialist",
            goal="Find REAL security vulnerabilities with exploit examples",
            backstory="You find actual vulnerabilities with file:line, exploit scenarios, and fixes.",
            llm=llm,
            verbose=True,
        )
        
        perf_agent = Agent(
            role="Performance Engineer",
            goal="Find measurable performance bottlenecks",
            backstory="You identify specific bottlenecks with measurements and optimized code.",
            llm=llm,
            verbose=True,
        )
        
        doc_agent = Agent(
            role="Documentation Expert",
            goal="Find specific documentation gaps",
            backstory="You list specific functions missing docs and provide example docstrings.",
            llm=llm,
            verbose=True,
        )
        
        # Define tasks
        progress(0.6, desc="📝 Creating analysis tasks...")
        
        arch_task = Task(
            description=f"""Find SPECIFIC architecture issues with exact locations.

{code_stats}

STRUCTURE:
{repo_structure}

CODE:
{code_summary}

Provide:
- Exact file:line for each issue
- Current code snippets
- Recommended improvements with code
- Impact analysis""",
            expected_output="Architecture review with 5+ specific issues and code examples",
            agent=arch_agent,
        )
        
        quality_task = Task(
            description=f"""Find SPECIFIC code quality issues.

{code_summary}

List:
- Complex functions (name + location)
- Code duplication (show both locations)
- Poor naming (list specific names)
- Long functions to split

Provide before/after code.""",
            expected_output="Code quality report with 5+ issues and refactoring examples",
            agent=quality_agent,
        )
        
        security_task = Task(
            description=f"""Find REAL security vulnerabilities.

{code_summary}

Find:
- SQL injection (show queries)
- XSS (show unsafe code)
- Auth issues (point to functions)
- Hardcoded secrets

Format: File:Line, Vulnerable code, Exploit, Fix, CVE/CWE""",
            expected_output="Security audit with 3+ vulnerabilities and fixes",
            agent=security_agent,
        )
        
        perf_task = Task(
            description=f"""Find SPECIFIC performance bottlenecks.

{code_summary}

Identify:
- N+1 queries (show ORM calls)
- Inefficient algorithms (Big-O)
- Missing indexes
- Blocking I/O

Provide measurements (before/after).""",
            expected_output="Performance analysis with 3+ bottlenecks and metrics",
            agent=perf_agent,
        )
        
        doc_task = Task(
            description=f"""Find SPECIFIC documentation gaps.

{code_summary}

List:
- Functions missing docstrings (by name)
- Complex code needing comments
- Missing type hints

Provide example docstrings.""",
            expected_output="Documentation review with 5+ gaps and examples",
            agent=doc_agent,
        )
        
        # Create crew
        progress(0.7, desc="⚡ Running analysis (this takes 3-5 minutes)...")
        
        crew = Crew(
            agents=[arch_agent, quality_agent, security_agent, perf_agent, doc_agent],
            tasks=[arch_task, quality_task, security_task, perf_task, doc_task],
            process=Process.sequential,
            verbose=True,
        )
        
        # Execute
        result = crew.kickoff()
        
        progress(0.95, desc="📋 Formatting report...")
        
        # Format report
        report = f"""
{'='*80}
            🤖 COMPREHENSIVE AI CODE REVIEW REPORT
{'='*80}

📦 Repository: {repo_url}
📊 Files Analyzed: {len(code_files)}
📝 Lines of Code: {sum(len(c.split('\\n')) for c in code_files.values()):,}
🤖 AI Model: gpt-4.1-nano

{'='*80}
                    📁 REPOSITORY STRUCTURE
{'='*80}

{repo_structure}

{'='*80}
                    📊 CODE STATISTICS
{'='*80}

{code_stats}

{'='*80}
                🔍 DETAILED ANALYSIS RESULTS
{'='*80}

{result}

{'='*80}
                        ✅ END OF REPORT
{'='*80}

💡 This report includes specific file locations, code examples, and 
   actionable recommendations from 5 specialized AI agents.
"""
        
        progress(1.0, desc="✅ Analysis complete!")
        return report
        
    except Exception as e:
        import traceback
        error = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        error += "\n\nCheck:\n- Repository URL is correct and public\n"
        error += "- OpenAI API key in .env file\n- API credits available"
        return error
    
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


# Create Gradio interface
def create_interface():
    """Create the Gradio web interface."""
    
    with gr.Blocks(
        title="AI Code Review System",
        theme=gr.themes.Soft(),
        css=".gradio-container {max-width: 1200px !important}"
    ) as interface:
        
        gr.Markdown("""
        # 🤖 AI-Powered Deep Code Review System
        
        Get comprehensive code reviews from **5 specialized AI agents** using **gpt-4.1-nano**:
        - 🏗️ Architecture Reviewer
        - 🔍 Code Quality Analyst  
        - 🔒 Security Auditor
        - ⚡ Performance Optimizer
        - 📚 Documentation Specialist
        
        ### How to use:
        1. Enter a public GitHub repository URL
        2. Click "Start Deep Code Review"
        3. Wait 3-5 minutes for comprehensive analysis
        """)
        
        with gr.Row():
            repo_input = gr.Textbox(
                label="📦 GitHub Repository URL",
                placeholder="https://github.com/username/repository",
                lines=1,
                scale=4
            )
        
        with gr.Row():
            analyze_btn = gr.Button(
                "🚀 Start Deep Code Review",
                variant="primary",
                size="lg",
                scale=2
            )
            clear_btn = gr.Button(
                "🗑️ Clear",
                variant="secondary",
                size="lg",
                scale=1
            )
        
        output_box = gr.Textbox(
            label="📊 Analysis Results",
            lines=30,
            max_lines=50,
            show_copy_button=True,
            placeholder="Analysis results will appear here...\n\nThis includes:\n- Exact file locations and line numbers\n- Security vulnerabilities with exploits\n- Performance bottlenecks with measurements\n- Code quality issues with refactoring examples\n- Documentation gaps with example docstrings"
        )
        
        with gr.Accordion("📝 Example Repositories", open=False):
            gr.Markdown("""
            **Quick Examples (2-3 min):**
            - `https://github.com/psf/requests` - HTTP library
            - `https://github.com/pallets/click` - CLI tool
            
            **Medium Examples (4-5 min):**
            - `https://github.com/pallets/flask` - Web framework
            - `https://github.com/fastapi/fastapi` - API framework
            """)
        
        with gr.Accordion("ℹ️ About This System", open=False):
            gr.Markdown("""
            ### What You Get:
            - **Specific Issues**: Exact file names and line numbers
            - **Code Examples**: Before/after code snippets
            - **Security Details**: CVE/CWE references, exploit scenarios
            - **Performance Metrics**: Measurements and impact analysis
            - **Actionable Fixes**: Concrete code improvements
            
            ### Technical Details:
            - **Model**: gpt-4.1-nano (fast and accurate)
            - **Agents**: 5 specialized AI reviewers
            - **Analysis Time**: 3-5 minutes per repository
            - **Cost**: ~$0.01-0.05 per analysis
            """)
        
        # Event handlers
        def validate_and_analyze(url):
            if not url or not url.strip():
                return "⚠️ Please enter a GitHub repository URL."
            if not url.startswith("https://github.com/"):
                return "⚠️ Please enter a valid GitHub URL (must start with https://github.com/)"
            return analyze_repository(url)
        
        analyze_btn.click(
            fn=validate_and_analyze,
            inputs=[repo_input],
            outputs=[output_box]
        )
        
        clear_btn.click(
            fn=lambda: ("", ""),
            inputs=None,
            outputs=[repo_input, output_box]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["https://github.com/psf/requests"],
                ["https://github.com/pallets/flask"],
                ["https://github.com/fastapi/fastapi"],
            ],
            inputs=[repo_input],
            label="Click an example to auto-fill"
        )
    
    return interface


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 AI Code Review System - Starting...")
    print("=" * 70)
    print("\n✨ Features:")
    print("   - Model: gpt-4.1-nano")
    print("   - 5 specialized AI agents")
    print("   - Specific file locations and code examples")
    print("   - Security, performance, and quality analysis")
    print("\n📱 Opening Gradio interface...")
    print("=" * 70)
    print()
    
    # Create and launch
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )