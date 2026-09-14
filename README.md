# Workshop tutorial on Shape Metrics using the IBL task

In this tutorial, we will explore a question: 

**Is geometrical structure shared across different brain regions?** 

Using recordings from the International Brain Laboratory brain-wide map (mice performing a decision-making task), we'll treat each brain region as a "system" and compare them with the shape-metrics recipe:

1. **Construct** a representation matrix for each region (conditions x neurons).
2. **Compare** matrices pairwise with the Procrustes distance.
3. **Analyse** the resulting distance matrix — does it relate to things we
   already know about these regions, like anatomical connectivity or their
   position in the cortical hierarchy?

## Setup

```bash
python -m venv ibl
# Windows
C:\Users\Your Name> myfirstproject\Scripts\activate
# macOS/Linux
source ibl/bin/activate

pip install -r requirements.txt
```
