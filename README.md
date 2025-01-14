# Bachelor Thesis Jesse Nieland

This is the README files of the Github repository for the code and files I created. This includes multiple Python, CSV and PNG files. Quite a few of these files have been moved and might not run properly anymore. This file should give a quick walkthrough as to what does work.

## Important Python Files (which don't have to be run anymore)

### vlr_scraper_all.py

This is the webscraper. It has a rate limit set in place and gets its data from [VLR](https://www.vlr.gg/). It gathers the data from the 2023 and 2024 VCT seasons.

### 2324_cleaner.py

This files cleans the data collected through the web scraper. Currently it takes manually changed CSV files, as some errors had to be fixed manually. It returns CSV files that are formatted in a way to make future work easier.

### visualization.py

This code visualizes interesting stastical analysis. All the analysis is saved as PNGs in the same folder. 

### data_preparation.py

This file prepares the data for the heuristics.py and model_halftime.py files. Previously, the same thing was done in both files, so it has been moved into one.

## Runnable Files

### heuristics.py

This file runs the halftime heuristic and returns a few different accuracies. It gives a picture of the strength of the heuristic. Some accuracy calculations are done on a subset of the dataset.

### model_halftime.py

This is the machine learning code. A few features are created, after which multiple models are run. Additionally, multiple performance metrics are provided. Lastly, some visualizations are added.

You can change some parameters up. There's code commented that changes the dataset to only include tied halftime scores. Additionally, features can be added and removed, and parameters for models can be tweaked. These are not defined globally and have to be edited at their specific lines.
