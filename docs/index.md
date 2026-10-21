# Artificial Neural Networks and Deep Learning

**Deena El Orra**, Insper 2026.2. Instructor: Humberto Sandmann. GitHub: [DeenaElOrra](https://github.com/DeenaElOrra)

Portfolio for the course. Each exercise lives in `docs/exercises/<slug>/`, with the report (`index.md`), the code that was run (`code/`) and the figures it shows (`figures/`).

## Exercises

| Exercise | Report |
|---|---|
| 2. Perceptron | [exercises/perceptron](exercises/perceptron/index.md) |

## Reproducing

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python docs/exercises/perceptron/code/perceptron.py   # figures + results.json
mkdocs serve                                           # local preview
```
