# Note : This file is not essintial and just an utility to fix error in the file naming
# Ignore the file in future use.
import os
# Fix the gloss numbering
glosses = ["A", "B", "C", "D", "E"]
for gloss in glosses:
    for i in range(1, 21):
        if os.path.isfile(f"./{gloss}/{i}.csv"):
            os.rename(f"./{gloss}/{i}.csv", f"./{gloss}/{i-1}.csv")
