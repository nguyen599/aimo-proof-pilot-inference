# Verifier audit replay - strict aggregation

| Problem | Old score | New score | Status | Low cap | Fatal cap | Role scores |
| --- | ---: | ---: | --- | --- | --- | --- |
| 1 | 0.625 | 0.375 | validated_low_score | False | False | dependency=0.0, counterexample=1.0, quantifier_algebra=0.5, coverage=0.5 |
| 2 | 1.0 | 0.875 | weighted_score_pass | False | False | dependency=1.0, counterexample=1.0, quantifier_algebra=1.0, coverage=1.0 |
| 3 | 0.5625 | 0.375 | validated_low_score | False | False | dependency=0.5, counterexample=0.5, quantifier_algebra=0.5, coverage=0.5 |
| 4 | 1.0 | 1.0 | strict_pass | False | False | dependency=1.0, counterexample=1.0, quantifier_algebra=1.0, coverage=1.0 |
| 5 | 1.0 | 0.5 | validated_low_score | True | False | dependency=0.5, counterexample=1.0, quantifier_algebra=1.0, coverage=1.0 |
| 6 | 1.0 | 0.5 | weighted_score_low | False | False | dependency=0.0, counterexample=1.0, quantifier_algebra=1.0, coverage=0.0 |
