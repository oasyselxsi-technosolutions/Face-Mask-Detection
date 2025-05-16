## To see only the currently installed Python versions using pyenv, run:

```
pyenv versions
```

This will list all Python versions installed via pyenv. The currently active version will be marked with an asterisk (*).
pyenv versions



pyenv local 3.9.21

This repo uses the Python 3.9.21

### Steps to Create a Local Python Environment

1. **Install the Desired Python Version**:
   - List available Python versions:
     ```powershell
     pyenv install --list
     ```
   - Install a specific Python version (e.g., 3.12.0):
     ```powershell
     pyenv install 3.12.0
     ```

2. **Set the Local Python Version for Your Project**:
   - Navigate to your project directory:
     ```powershell
     cd path\to\your\django\project
     ```
   - Set the local Python version:
     ```powershell
     pyenv local 3.12.0
     ```

3. **Create a Virtual Environment**:
   - Create a virtual environment using the specified Python version:
     ```powershell
     python -m venv venv
     ```

4. **Activate the Virtual Environment**:
   - On Windows:
     ```powershell
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```sh
     source venv/bin/activate
     ```

5. **Install Dependencies from `requirements.txt`**:
   ```powershell
   pip3 install -r requirements.txt
   ```
