# The Process

This file was used to keep track of progress on the project and identify tasks.

Each model should take as input a config file?

#TODO: Investigate if the baseline_cnn model accuracy is actually correct and why it's so high #DONE

#TODO: use a proper metrics script actually do fair evaluation and comparison between models #DONE

#TODO: Make saliency map and analyze it to ensure model fairness #DONE

#TODO: Look at model false negs adn false pos of model predictions

#TODO: Start saving models! and experiment data! #IN PROGRESS

#TODO review literature and implement "SOTA" model #DONE

Next steps:

1. review input data transformations and implement (make importable (e.g a single .py file)) #DONE
2. identify all possible hyperparamets #DONE
3. setup random search/quasi-random search (if not too hard) infrastructure in each experimental.py file for hyperparameter optimization #DONE
4. come up with setup to collect outputs and loss values for charts and comparisons. #DONE
5. Put everything in bash scripts and make a bashscript that will submit the jobs. #DONE
6. Analyze results, identify best models and their hyperparameters and train them for longer. #PENDING

Param estimates
efficient_b0 — 5.3M
efficient_b4— 19.3M
mesonet — ~28K
vision_trans — 86.6M
xception — 22.9M
