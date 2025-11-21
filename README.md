# Polymarket Arbitrage Opportunity Detection

This repo was created for Rice University's STAT 482 final project by Group 1.

## Installation

After cloning and `cd`'ing into the repo, create and activate a Python virtual environment. Example for virtualenv:

Linux/MacOS:
```
python3 -m venv venv
source venv/bin/activate
```

Windows Command Prompt:
```
py -m venv venv
.\venv\Scripts\activate.bat
```

Windows PowerShell:
```
py -m venv venv
.\venv\Scripts\Activate.ps1
```

Install requirements using
```
pip install -r requirements.txt
```

## Usage
`backtest.py` is the script used for backtesting the algorithms.

`execute_trades.py` is the script used for using the algorithms to execute real time trades and hopefully not lose money.

`generate_presentation_materials.py` is the Jupyter-like script used for interactive programming in VSCode, which generates the presentation materials needed for the STAT 482 final presentation.
