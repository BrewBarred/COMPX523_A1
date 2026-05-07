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

""" Import the necessary libraries """
# import the capymoa "Electricity" dataset
from capymoa.datasets import Electricity
# import our preferred classifier
from assignment1_1631819 import KNN_cw_1631819 as clsf

""" Test to ensure capymoa and custom classifer works """
stream = Electricity()
print(f'Loaded stream: "{stream}"')
schema = stream.get_schema()
print(f'Fetched schema:\n{schema}')
classifier = clsf(schema=schema, k=3, w=500)
classifier_name = clsf.__name__
print(f'\nDetected classifier: "{classifier_name}"')

""" Setup the stream vars """
correct = 0
# store unique stream labels
labels = set()
# count instances
i = 0

""" Start training on/predicting datastream """
print(f'Streaming, classifying and training on "{stream}" data-stream...')
# iterate through the whole stream
while stream.has_more_instances():
    # track instance count
    i+=1

    # fetch the next instance in the stream
    instance = stream.next_instance()
    # use our nominated classifier to predict the label of this instnace
    prediction = classifier.predict(instance)
    # if the label is correct, add it to the prediciton count
    if prediction == instance.y_index:
        correct+=1
    # train the classifier further
    classifier.train(instance)

    # determine the label of this instance
    label = instance.y_index
    # add unique labels to our set (since it's a set, duplicates are automatically ignored)
    labels.add(label)

""" Print the results """
print(f'Found {len(labels)} unique labels {labels} in the "{stream}" data-stream.')
print(f'{classifier_name} accuracy: {correct/i*100:.1f}% ({i} instances)')
print(f'Class windows: {list(classifier.class_window.keys())}')
print(f'Window sizes: {[classifier.class_window[k].size() for k in classifier.class_window]}')