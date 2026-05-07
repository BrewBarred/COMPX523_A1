"""
Activate the capymoa python virtual environment:

    cd ~/compx523

    source venv/bin/activate

Optional, but recommended:
# store git credentials to avoid having to enter them every time you push to github

    git config --global credential.helper store

# push to lock in

    git push -u origin main
"""

"""
Import the necessary libraries
"""
import capymoa
# confirm capyMOA was successfully imported
print(f'CapyMOA version {capymoa.__version__} detected.')
from capymoa.datasets import Electricity
# confirm Electricity dataset was successfully imported
print(f'Loaded dataset: "{Electricity()}"')
# confirm classifier can be imported from our custom file
from assignment1_1631819 import KNN_cw_1631819 as classifier
print(f'Detected classifier: "{classifier}"')

# run a test script to verify that CapyMOA is works
stream = Electricity()
schema
# store unique stream labels
labels = set()
# count instances
i = 0
# iterate through the whole stream
while stream.has_more_instances():
    # track instance count
    i+=1
    # fetch the next instance in the stream
    instance = stream.next_instance()
    # determine the label of this instance
    label = instance.y_index
    # add unique labels to our set (since it's a set, duplicates are automatically ignored)
    labels.add(label)

# 4. Print the results
print(f'Found {len(labels)} unique labels {labels} in the "{stream}" data-stream.')