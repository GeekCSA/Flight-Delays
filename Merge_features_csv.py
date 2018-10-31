import numpy as np
import pandas as pd

left = pd.read_csv("dataset/Medium_Flight.csv")
right = pd.read_csv("dataset/NOAA_LAX_MIA.csv")
result = pd.merge(left, right, how='inner', on=['DATE', 'NAME'])

print(np.array(result))

df = pd.DataFrame(result)
df.to_csv("RESULT_LAX_MIA.csv")