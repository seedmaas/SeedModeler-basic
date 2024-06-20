# Automatic Modeling : Automated Optimization Code Solutions for ICML 2024

Welcome to the official code repository for our submission to the ICML 2024 Challenges on Automated Math Reasoning, specifically for Track 3: Automated Optimization Problem-Solving with Code.

## Overview

This repository contains the Python code and associated documentation necessary to understand and replicate our successful approach to the competition's challenges. Our solution leverages advanced algorithms and mathematical frameworks to automatically generate and solve optimization problems, pushing the boundaries of automated mathematical reasoning.


## Getting Started

## System Requirements

- Python 3.9 or higher

## Installation Steps

1. **Install Python**

Ensure that Python 3.9 or a newer version is installed on your system. You can check your Python version by running:

```shell
python --version
```

If Python is not installed, download and install it from the [official Python website](https://www.python.org/downloads/).

2. **Configure Virtual Environment (Optional)**

To avoid any conflicts with other Python projects on your system due to dependencies, it's recommended to configure a virtual environment for this project. Use the following commands to create and activate a virtual environment:

```shell
python -m venv venv
source venv/bin/activate  # On Unix or MacOS
.\venv\Scripts\Activate   # On Windows
```

3. **Install Dependencies**

There is a `requirements.txt` file located in the project directory that lists all the necessary Python libraries for the project. Install these libraries using:

```shell
pip install -r requirements.txt
```

4. **Configure OS Environment Variables**

This project requires specific environment variables to be set for correct operation. Instead of manually setting these variables in your OS, you can configure them by using a `.env` file as a reference. 

- First, locate the `.env.example` file in the project's root directory. This file contains a list of all the necessary environment variables with placeholder values.
  
- Create a new file named `.env` in the same directory.

- Open the `.env.example` file, and for each variable, replace the placeholder values with your actual data. Then, copy these settings into your newly created `.env` file.

5. **Place Data Files**

Ensure your data files (in .json format) are placed within the project's directory. These files are essential for the project's execution.

## Running the Project

Once all the above steps are completed, you can run the `main.py` file with the following command:

```shell
python main.py
```
---

Please note, the `ICML_test_no_optimal.py` script includes comprehensive comments and instructions on usage and customization options.

## Repository Structure

- `README.md` - This file, containing an overview and setup instructions.
- `main.py` - The primary script for executing our model.
- `utils/` `agent/`- Contains Python scripts defining the optimization workflow used.
- `PromptEngineering/` - Directory for storing input prompt.


