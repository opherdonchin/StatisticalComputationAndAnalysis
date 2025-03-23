import os
import nbformat
from nbconvert import PythonExporter
import argparse
import sys
import glob

def convert_py_to_ipynb(file_path):
    """
    Convert a Python script (.py) to a Jupyter notebook (.ipynb).
    
    Args:
        file_path: Path to the Python script to convert
    
    Returns:
        str: Path to the created notebook file
    """
    # Read the Python file
    with open(file_path, 'r', encoding='utf-8') as f:
        python_code = f.read()
    
    # Create a new notebook
    notebook = nbformat.v4.new_notebook()
    
    # Add the Python code as a cell
    cell = nbformat.v4.new_code_cell(python_code)
    notebook.cells.append(cell)
    
    # Save the notebook
    notebook_filename = os.path.splitext(file_path)[0] + '.ipynb'
    with open(notebook_filename, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
    
    return notebook_filename


def convert_ipynb_to_py(file_path):
    """
    Convert a Jupyter notebook (.ipynb) to a Python script (.py).
    
    Args:
        file_path: Path to the Jupyter notebook to convert
    
    Returns:
        str: Path to the created Python script file
    """
    try:
        # Convert notebook to Python script
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Convert the notebook to a Python script
        exporter = PythonExporter()
        script, _ = exporter.from_notebook_node(notebook)
        
        # Save the Python script
        script_filename = os.path.splitext(file_path)[0] + '.py'
        with open(script_filename, 'w', encoding='utf-8') as f:
            f.write(script)
        
        return script_filename
    except Exception as e:
        print(f"Error converting {file_path}: {str(e)}")
        return None


def process_files(path, to_python=True):
    """
    Find files and convert between Python scripts and Jupyter notebooks.
    
    Args:
        path: Path to file(s), may include wildcards
        to_python: If True, convert .ipynb to .py; if False, convert .py to .ipynb
    
    Returns:
        int: 0 for success, 1 for failure
    """
    # Determine file extension to look for
    extension = '.ipynb' if to_python else '.py'
    
    # Find files to process
    if os.path.isfile(path) and path.endswith(extension):
        # Single file
        file_paths = [path]
    else:
        # Treat as pattern with possible wildcards
        file_paths = glob.glob(path)
        # Filter to only include files with the right extension
        file_paths = [f for f in file_paths if f.endswith(extension)]
    
    if not file_paths:
        print(f"Error: No files found matching '{path}' with extension '{extension}'.")
        return 1
    
    converted_files = []
    count = 0
    
    for file_path in file_paths:
        # Call the appropriate conversion function
        if to_python:
            result = convert_ipynb_to_py(file_path)
        else:
            result = convert_py_to_ipynb(file_path)
        
        if result:
            converted_files.append(result)
            count += 1
    
    if count > 0:
        print(f"Conversion complete. Converted {count} file(s):")
        for file in converted_files:
            print(f" - {file}")
        return 0
    else:
        print(f"No files were converted.")
        return 1

def main():
    parser = argparse.ArgumentParser(description='Convert between Jupyter notebooks and Python scripts.')
    parser.add_argument('path', help='Path to file(s), may include wildcards')
    parser.add_argument('--to-py', '--to-python', dest='to_python', action='store_true', 
                        help='Convert .ipynb files to .py (default)')
    parser.add_argument('--to-ipynb', '--to-notebook', dest='to_notebook', action='store_true',
                        help='Convert .py files to .ipynb')
    
    args = parser.parse_args()
    
    # Determine direction
    if args.to_notebook and args.to_python:
        parser.error("Cannot specify both --to-py and --to-ipynb at the same time.")
    
    # Default is to convert to Python
    to_python = not args.to_notebook
    
    return process_files(args.path, to_python=to_python)


if __name__ == "__main__":
    main()
