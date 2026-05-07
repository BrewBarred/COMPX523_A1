from capymoa.base import Classifier
from moa.classifiers.lazy.neighboursearch import KDTree
from moa.classifiers.lazy.neighboursearch import LinearNNSearch
from com.yahoo.labs.samoa.instances import Instances
import math

##############################################################
####################### HELPER WINDOW ########################
##############################################################
class newWindow():
    def __init__(self, schema, instances=None):
        self.window = Instances(schema.get_moa_header(), 0)
        self.schema = schema
        if instances is not None:
            for i in range(instances.numInstances()):
                self.window.add(instances.get(i))
    
    def add_instance(self, instance):
        self.window.add(instance.java_instance.getData())

    def add_window(self, nwindow):
        rw = nwindow.get_window()
        for i in range(nwindow.size()):
            self.window.add(rw.get(i))

    def remove_instance(self, position=0):
        self.window.delete(position)

    def get_instance(self, position=0):
        return self.window.get(position)

    def get_window(self):
        return self.window
    
    def get_schema(self):
        return self.schema
    
    def size(self):
        return self.window.numInstances()
    
##############################################################
###################### HELPER NNSEARCH #######################
##############################################################
class SearchNeighbours():
    def __init__(self, search_method):
        assert search_method in ["LinearNNSearch", "KDTree"]
        self.search_method_str = search_method
        if self.search_method_str == "LinearNNSearch":
            self.search_method = LinearNNSearch()
        else:
            self.search_method = KDTree()

    def do_search(self, window, instance, arg_k):
        self.search_method.setInstances(window.get_window())
        k = min(arg_k, window.size())
        neighbors = self.search_method.kNearestNeighbours(instance.java_instance.getData(), k)
        return newWindow(window.get_schema(), neighbors)
    
##############################################################
###################### HELPER DISTANCE #######################
##############################################################
def distanceMath(instance1, instance2):
	if 'LabeledInstance' in type(instance1).__name__:
		inst1 = instance1.java_instance.getData().toDoubleArray()
	if 'LabeledInstance' in type(instance2).__name__:
		inst2 = instance2.java_instance.getData().toDoubleArray()
	elif 'InstanceImpl' in type(instance2).__name__:
		inst2 = instance2.toDoubleArray()
	return math.sqrt(sum([(inst1[i] - inst2[i])**2 for i in range(len(inst1)-1)]))
    
##############################################################
######################## KNN CLASSES #########################
##############################################################
class KNNBase(Classifier):
    def __init__(self, schema, k=3, window_size=1000, search_method="LinearNNSearch"):
        super().__init__(schema)
        self.k = k
        self.window_size = window_size
        self.search = SearchNeighbours(search_method)
        self.window = newWindow(schema)

    def __str__(self):
        return "Base kNN"

    def train(self, instance):
        if self.window.size() >= self.window_size:
            self.window.remove_instance()
        self.window.add_instance(instance)
    
    def predict(self, instance):
        """
        Returns the k-NN predictions for a given instance.
        Uses mean/median for regression or majority vote for classification.
        """
        try:
            if self.window and self.window.size() > 0:
                # Find k nearest neighbors
                neighbors = self.search.do_search(self.window, instance, self.k)
                # Vote for the most common class among the k closest neighbors
                votes = [0] * (self.schema.get_num_classes())
                for i in range(neighbors.size()):
                    class_idx = int(neighbors.get_instance(i).classValue())
                    votes[class_idx] += 1
                return votes.index(max(votes))

        except Exception as e:
            print("Error: kNN search failed.", e)
            return 0

    def predict_proba(self, instance):
        pass

class KNNBase3(Classifier):
    def __init__(self, schema, k=3, window_size=1000, search_method="LinearNNSearch"):
        super().__init__(schema)
        self.k = k
        self.window_size = window_size
        self.search = SearchNeighbours(search_method)
        self.window = newWindow(schema)

    def __str__(self):
        return "Base3 kNN"

    def train(self, instance):
        if self.window.size() >= self.window_size:
            self.window.remove_instance()
        self.window.add_instance(instance)
    
    def predict(self, instance):
        """
        Returns the k-NN predictions for a given instance.
        Uses mean/median for regression or majority vote for classification.
        """
        try:
            if self.window and self.window.size() > 0:
                # Find k nearest neighbors
                neighbors = self.search.do_search(self.window, instance, self.k)
                # Vote for the most common class among the k closest neighbors
                votes = [0] * (self.schema.get_num_classes())
                for i in range(neighbors.size()):
                    class_idx = int(neighbors.get_instance(i).classValue())
                    votes[class_idx] += 1
                return votes.index(max(votes))

        except Exception as e:
            print("Error: kNN search failed.", e)
            return 0

    def predict_proba(self, instance):
        pass

class KNN_cw_1631819(Classifier):
    def __init__(self, schema, k=3, w=500, search_method="LinearNNSearch"):
        super().__init__(schema)
        self.k = k
        self.w = w
        self.search = SearchNeighbours(search_method)
        self.class_window = {}  # dict: label_index -> newWindow

    def __str__(self):
        return "KNN_cw_1631819"

    def train(self, instance):
        label = instance.y_index
        if label not in self.class_window:
            self.class_window[label] = newWindow(self.schema)
        self.class_window[label].add_instance(instance)
        if self.class_window[label].size() > self.w:
            self.class_window[label].remove_instance(0)

    def predict(self, instance):
        if not self.class_window:
            return 0
        pool = []
        for label, window in self.class_window.items():
            if window.size() == 0:
                continue
            neighbours = self.search.do_search(window, instance, self.k)
            for i in range(neighbours.size()):
                n = neighbours.get_instance(i)
                dist = distanceMath(instance, n)
                pool.append((dist, label))
        if not pool:
            return 0
        pool.sort(key=lambda x: x[0])
        final_neighbours = pool[:self.k]
        votes = [0] * self.schema.get_num_classes()
        for dist, label in final_neighbours:
            votes[label] += 1
        return votes.index(max(votes))

    def predict_proba(self, instance):
        pass