#!/usr/bin/python

import random
import util
import re
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict
from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 1: binary classification
############################################################

############################################################
# Problem 1a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    pass
    # ### START CODE HERE ###

    arr = x.lower().split()  # Convert to lowercase and split
    featureMap = {} # Define an empty featureMap dictionary

    # Taking the value of all space seperated characters (Including Punctuations)
    # I was filtering the non alphanumeric elements before 
    # but keeping them in gives me more data and lower error rate
    for word in arr:
        featureMap[word] = 1 + featureMap.get(word, 0)

    return featureMap
    # ### END CODE HERE ###


############################################################
# Problem 1b: stochastic gradient descent

T = TypeVar("T")


def learnPredictor(
    trainExamples: List[Tuple[T, int]],
    validationExamples: List[Tuple[T, int]],
    featureExtractor: Callable[[T], FeatureVector],
    numEpochs: int,
    eta: float,
) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    """
    weights = {}  # feature => weight
    # ### START CODE HERE ###

    def getScore(features):
        score = 0
        for word, feature in features.items():
            score += weights.get(word,0) * feature
        
        # We expect this score to be negative for a negative review and positive for a positive review
        return score
                
    def updateWeights(features, expected_classification):

        gradients = {}
        for word, feature in features.items():
            gradients[word] = -expected_classification * feature

        # Update the weights
        for word, gradient in gradients.items():
            weights[word] = weights.get(word, 0) - eta * gradient

    for epoch in range(numEpochs):
        for review, expected_classification in trainExamples:
            features = featureExtractor(review)
            score = getScore(features)

            # Using Gradient Hinge Loss Expression -
            # Here expected_classification * score i.e. y * score is the margin.
            # Basically, we check if our margin is greater than 1. If it is greater than 1, our model is correct.
            # If margin is less than 1, our model is incorrect and we need to update weights.       
            if 1 - (expected_classification * score) > 0:                
                updateWeights(features, expected_classification)
            # No Else Condition requried as gradient would be 0 if margin was greater than 1            

    # ### END CODE HERE ###
    return weights


############################################################
# Problem 1c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:#
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        phi = None
        y = None
        # ### START CODE HERE ###

        subset_weights = random.sample(list(weights.keys()), random.randint(1, len(weights)))

        phi = {}
        for i in subset_weights:
            # Choosing equal positive and negative magnitudes to prevent weights from leaning more on one side
            phi[i] = random.uniform(-10, 10)

        # Calculating the new score based on arbitary feature values
        dot_product = util.dotProduct(phi, weights)
        
        # If score is positive, it is a positive classification (i.e. positive review)
        if dot_product >= 0:
            y = 1
        # If score is negative, it is a negative classification (i.e. negative review)
        else:
            y = -1

        # ### END CODE HERE ###
        return (phi, y)

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 1d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x):
        pass
        # ### START CODE HERE ###
    
        # Remove all spaces using regex
        review = re.sub(r'\s+', '', x) 
        featureMap = {} 
        
        start = 0
        while start + n <= len(review):

            # Using string slicing to extract n sized substrings
            featureMap[review[start: start + n]] = 1 + featureMap.get(review[start: start + n], 0)  
            start = start + 1

        return featureMap
    
        # ### END CODE HERE ###

    return extract


############################################################
# Problem 1e:
#
# Helper function to test 1e.
#
# To run this function, run the command from termial with `n` replaced
#
# $ python -c "from submission import *; testValuesOfN(n)"
#


def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be submitted.
    """
    trainExamples = readExamples("polarity.train")
    validationExamples = readExamples("polarity.dev")
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(
        trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01
    )
    # outputWeights(weights, "weights")
    outputErrorAnalysis(
        validationExamples, featureExtractor, weights, "error-analysis"
    )  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    validationError = evaluatePredictor(
        validationExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    print(
        (
            "Official: For N = %s, train error = %s, validation error = %s"
            % (n, trainError, validationError)
        )
    )


############################################################
# Problem 2b: K-means
############################################################


def kmeans(
    examples: List[Dict[str, float]], K: int, maxEpochs: int
) -> Tuple[List, List, float]:
    """
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    """
    # ### START CODE HERE ###
        
    # Here while calculating euclidean square distance, for optimization 
    # instead of doing subtraction and then squaring we use the conversion 
    # (a-b)^2 = a^2 - 2ab + b^2. This allows us to speed up the algorithm 
    # as we already have precomputed a^2 and b^2 using getSquares function below. 
    def get_distance(point1, point1_square, point2, point2_square):
        dot_product = util.dotProduct(point1, point2)
        return point1_square - 2 * dot_product + point2_square
    
    # We can calculate the square of a vector by taking a dot product with itself.
    def get_squares(input_arr):
        squares = []
        for val in input_arr:
            squares.append(util.dotProduct(val, val))
        return squares

    # We use this function to calculate the center of any input cluster
    def calculate_center(cluster):
        if not cluster: return {}
        
        cluster_length = len(cluster)
        center = {}

        # For each element in the cluster, we take the sum of all the values.
        # For the center, we take the average of these values by diving the sum by cluster length
        for element in cluster:
            for key, value in element.items():
                if key in center:
                    center[key] += value
                else:
                    center[key] = value

        for key in center:
            center[key] = center[key] / cluster_length
        return center

    # Randomly initialize K cluster centers to random elements of examples
    centers = random.sample(examples, K)

    # Creating the assignments array that will store the cluster assignment of each element at the 
    # same index as the index of the element in examples array. Intiializing assigning all elements to cluster 0
    assignments = [0] * len(examples)

    # Precomputing the squares of all example elements by taking their dotProduct with themseleves.
    examples_squares = get_squares(examples)
 
    for epoch in range(maxEpochs):

        centers_squares = get_squares(centers)

        for exampleIndex, example in enumerate(examples):

            # Calculate the distance of each point from the initial centers
            distances = []
            for centerIndex, center in enumerate(centers):
                distances.append(get_distance(example, examples_squares[exampleIndex], center, centers_squares[centerIndex]))

            # We assign the index of the centroid that is closest to each given example and manage this using example Index
            assignments[exampleIndex] = distances.index(min(distances))

        # Updating the centers
        new_centers = []
        # Here cluster_i represents each cluster in our total clusters K
        for cluster_i in range(K):
            cluster = []
            for exampleIndex, example in enumerate(examples):
                # Here we check if the current example is assigned cluster_i. 
                # If that is true, we group them all together in the newly created cluster array.
                if assignments[exampleIndex] == cluster_i:
                    cluster.append(example)

            new_centers.append(calculate_center(cluster))

        if new_centers == centers:  # Convergence check
            break
        centers = new_centers

    centers_squares = get_squares(centers)

    # Calculate total cost
    totalCost = 0
    for i in range(len(examples)):
        totalCost += get_distance(examples[i], examples_squares[i], centers[assignments[i]], centers_squares[assignments[i]])

    return centers, assignments, totalCost
    # ### END CODE HERE ###

# for i in range(50, 101):
#     testValuesOfN(i)
