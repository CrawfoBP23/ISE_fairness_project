# ISE Fairness Testing Project

This project evaluates individual fairness in a machine learning model by detecting Individual Discriminatory Instances (IDIs).

## Files

- `fairness_tool.py` – proposed improved fairness testing method  
- `lab4.py` – baseline random search method  
- `plot_graph.py` – generates IDI vs budget graph  

## Results

- `fairness_results.csv` – proposed method results  
- `baseline_results.csv` – baseline results  
- `wilcoxon_results.csv` – statistical test results  

## PDFs

- `requirements.pdf` – dependencies  
- `manual.pdf` – usage instructions  
- `replication.pdf` – how to reproduce results  

## How to Run

```bash
pip install -r requirements.txt
python fairness_tool.py
python lab4.py
python plot_graph.py
