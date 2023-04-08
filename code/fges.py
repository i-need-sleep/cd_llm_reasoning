from causallearn.search.ScoreBased.GES import ges
import json
import pandas as pd
import utils.globals as uglobals

with open(f'{uglobals.AGGREGATED_OUTPUT_DIR}/aggregated_5shot.json') as f:
    data = json.load(f)

def json_to_df(data):
    out = {}
    
    for model_idx, (model, results) in enumerate(data.items()):
        # out['model'].append(model_idx)
        # out['model'].append(model)
        for subject, score in results.items():
            if subject not in out.keys():
                out[subject] = []
            out[subject].append(float(score))
    out = pd.DataFrame(out)
    return out

len(data)
for model, scores in data.items():
    print(model, len(scores))
df = json_to_df(data)

# default parameters
Record = ges(df)

# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

pyd = GraphUtils.to_pydot(Record['G'])
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()


# or save the graph
pyd.write_png('simple_test.png')