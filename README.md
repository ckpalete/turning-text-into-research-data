# Turning Text into Research Data

### Measurement and inference with LLMs

Materials for the SAFE Data Seminar, [date], SAFE Research Data Center.

Large language models make it cheap to turn documents into variables — to read
thousands of filings, reports or disclosures and extract a measure you can analyse.
This seminar covers the step after that: checking whether the variable you created
is accurate enough to build a result on, and correcting the estimates if it is not.

## What is here

| File | |
|---|---|
| `turning_text_into_research_data.ipynb` | The seminar notebook. Start here. |
| `sim.py` | The simulation behind the figures in the slides. |
| `figs.py` | Generates the figures from `sim.py`. |
| `figures/` | The figures as PNG and PDF. |
| `requirements.txt` | Package versions. |

## Before the seminar

Install Python and an editor that can run Jupyter notebooks:

- Python: https://www.python.org/downloads/
- VSCodium: https://vscodium.com/ — or any alternative that runs notebooks

Then install the packages:

```
pip install numpy pandas matplotlib
```

No API key, no internet connection and no paid model access are required. Everything
in the notebook runs locally in about ten seconds.

If your setup does not work on the day, you can still follow along — every cell is
run on screen during the seminar.

## Running the notebook

Open `turning_text_into_research_data.ipynb` and run the cells in order. There are
three exercises; solutions to the first two are in the cells immediately following
them, and the third is a take-home.

To regenerate the figures used in the slides:

```
pip install statsmodels
python figs.py
```

## What the notebook covers

1. A corpus of 4,000 documents, labeled by two different prompts for the same concept
2. Label quality against hand-labeled ground truth — the diagnostic usually reported
3. The naive regression, and why both prompts give the wrong answer
4. Why the standard attenuation argument does not apply here
5. The correction: measure the bias on a hand-labeled sample and subtract it
6. Coverage, checked by simulation
7. When using a model is not worth it
8. A protocol for applying this to your own corpus

## A note on the data

**The corpus in the notebook is simulated, not real.** This is deliberate: because
the data is generated, the true coefficient is known exactly (β = 2.0), so every
estimator can be compared against the right answer. With real documents you never
get to check.

Section 8 of the notebook covers what changes when you bring your own corpus.

One number in the notebook deserves a warning. The break-even label accuracy of
roughly 78% in section 7 is **not a universal constant** — it follows from this
sample size, this number of hand labels, this effect size and this base rate. What
generalises is that a crossover exists, and that below it a poor label costs
precision rather than buying it. Find yours with the code before relying on a number.

## References

- Ludwig, J., Mullainathan, S. and Rambachan, A. (2026). "Large Language Models: An
  Applied Econometric Framework." *Annual Review of Economics* 18, 283–316.
  Ungated as NBER Working Paper 33344.
- Dell, M. and Rambachan, A. (2026). "Estimation and Inference with AI-Generated
  Data." NBER Summer Institute Methods Lectures. Video and slides at nber.org.
- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I. and Zrnic, T. (2023).
  "Prediction-Powered Inference." *Science* 382(6671), 669–674.

## Contact

[Your name] · [your email]
