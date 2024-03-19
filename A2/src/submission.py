from typing import List, Tuple

from mapUtil import (
    CityMap,
    computeDistance,
    createStanfordMap,
    locationFromTag,
    makeTag,
)
from util import Heuristic, SearchProblem, State, UniformCostSearch


# *IMPORTANT* :: A key part of this assignment is figuring out how to model states
# effectively. We've defined a class `State` to help you think through this, with a
# field called `memory`.
#
# As you implement the different types of search problems below, think about what
# `memory` should contain to enable efficient search!
#   > Check out the docstring for `State` in `util.py` for more details and code.

########################################################################################
# Problem 1a: Modeling the Shortest Path Problem.


class ShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path
    from `startLocation` to any location with the specified `endTag`.
    """

    def __init__(self, startLocation: str, endTag: str, cityMap: CityMap):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

    def startState(self) -> State:
        pass
        # ### START CODE HERE ###
        return State(self.startLocation, None)
        # ### END CODE HERE ###

    def isEnd(self, state: State) -> bool:
        pass
        # ### START CODE HERE ###
        return self.endTag in self.cityMap.tags.get(state.location, [])
        # ### END CODE HERE ###

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        pass
        # ### START CODE HERE ###
        result = []
        # Assuming self.cityMap.distances[state.location] correctly gives adjacent locations and distances
        for adjacentLocation, distance in self.cityMap.distances[state.location].items():
            result.append((adjacentLocation, State(adjacentLocation, None), distance))
        return result
        # ### END CODE HERE ###


########################################################################################
# Problem 1b: Custom -- Plan a Route through Stanford


def getStanfordShortestPathProblem() -> ShortestPathProblem:
    """
    Create your own search problem using the map of Stanford, specifying your own
    `startLocation`/`endTag`. If you prefer, you may create a new map using via
    `createCustomMap()`.

    Run `python mapUtil.py > readableStanfordMap.txt` to dump a file with a list of
    locations and associated tags; you might find it useful to search for the following
    tag keys (amongst others):
        - `landmark=` - Hand-defined landmarks (from `data/stanford-landmarks.json`)
        - `amenity=`  - Various amenity types (e.g., "park", "food")
        - `parking=`  - Assorted parking options (e.g., "underground")
    """
    cityMap = createStanfordMap()

    # Or, if you would rather use a custom map, you can uncomment the following!
    # cityMap = createCustomMap("data/custom.pbf", "data/custom-landmarks".json")

    startLocation, endTag = None, None

    # ### START CODE HERE ###
    startLocation = "landmark=oval"
    endTag = "amenity=cafe"
    # ### END CODE HERE ###
    return ShortestPathProblem(startLocation, endTag, cityMap)


########################################################################################
# Problem 2a: Modeling the Waypoints Shortest Path Problem.


class WaypointsShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path from
    `startLocation` to any location with the specified `endTag` such that the path also
    traverses locations that cover the set of tags in `waypointTags`.

    Think carefully about what `memory` representation your States should have!
    """
    def __init__(
        self, startLocation: str, waypointTags: List[str], endTag: str, cityMap: CityMap
    ):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

        # We want waypointTags to be consistent/canonical (sorted) and hashable (tuple)
        self.waypointTags = tuple(sorted(waypointTags))

    def startState(self) -> State:
        pass
        # ### START CODE HERE ###
        return State(location=self.startLocation, memory=self.waypointTags)
        # ### END CODE HERE ###

    def isEnd(self, state: State) -> bool:
        pass
        # ### START CODE HERE ###
        if state.memory is None or len(state.memory) == 0:
            return self.cityMap.getTag(state.location) == self.endTag
        return False
        # ### END CODE HERE ###

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        pass
        # ### START CODE HERE ###
        successors = []
        currentLocation = state.location
        remainingWaypoints = state.memory if state.memory is not None else ()

        # Access the 'distances' dictionary directly to find neighbors
        if currentLocation in self.cityMap.distances:
            for neighbor, cost in self.cityMap.distances[currentLocation].items():
                # Check if visiting 'neighbor' covers any waypoint tags
                neighborTags = self.cityMap.tags.get(neighbor, [])
                newWaypoints = tuple(tag for tag in remainingWaypoints if all(makeTag(tagType, tag) not in neighborTags for tagType in ["amenity", "landmark"]))

                # Form the new state
                newState = State(location=neighbor, memory=newWaypoints)

                # Action (in this case, can be represented by the neighbor's name), New State, Cost
                successors.append((neighbor, newState, cost))

        return successors
        # ### END CODE HERE ###

        
########################################################################################
# Problem 2b: Custom -- Plan a Route with Unordered Waypoints through Stanford


def getStanfordWaypointsShortestPathProblem() -> WaypointsShortestPathProblem:
    """
    Create your own search problem using the map of Stanford, specifying your own
    `startLocation`/`waypointTags`/`endTag`.

    Similar to Problem 1b, use `readableStanfordMap.txt` to identify potential
    locations and tags.
    """
    cityMap = createStanfordMap()

    startTag = None
    startLocation = None
    waypointTags = None
    endTag = None

    # ### START CODE HERE ###
    # ### END CODE HERE ###
    return WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap)


########################################################################################
# Problem 3a: A* to UCS reduction

# Turn an existing SearchProblem (`problem`) you are trying to solve with a
# Heuristic (`heuristic`) into a new SearchProblem (`newSearchProblem`), such
# that running uniform cost search on `newSearchProblem` is equivalent to
# running A* on `problem` subject to `heuristic`.
#
# This process of translating a model of a problem + extra constraints into a
# new instance of the same problem is called a reduction; it's a powerful tool
# for writing down "new" models in a language we're already familiar with.


def aStarReduction(problem: SearchProblem, heuristic: Heuristic) -> SearchProblem:
    class NewSearchProblem(SearchProblem):
        def startState(self) -> State:
            pass
            # ### START CODE HERE ###
            # ### END CODE HERE ###

        def isEnd(self, state: State) -> bool:
            pass
            # ### START CODE HERE ###
            # ### END CODE HERE ###

        def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
            pass
            # ### START CODE HERE ###
            # ### END CODE HERE ###

    return NewSearchProblem()


########################################################################################
# Problem 3b: "straight-line" heuristic for A*


class StraightLineHeuristic(Heuristic):
    """
    Estimate the cost between locations as the straight-line distance.
        > Hint: you might consider using `computeDistance` defined in `mapUtil.py`
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap

        # Precompute
        # ### START CODE HERE ###
        # ### END CODE HERE ###

    def evaluate(self, state: State) -> float:
        pass
        # ### START CODE HERE ###
        # ### END CODE HERE ###


########################################################################################
# Problem 3c: "no waypoints" heuristic for A*


class NoWaypointsHeuristic(Heuristic):
    """
    Returns the minimum distance from `startLocation` to any location with `endTag`,
    ignoring all waypoints.
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        """
        Precompute cost of shortest path from each location to a location with the desired endTag
        """
        # Define a reversed shortest path problem from a special END state
        # (which connects via 0 cost to all end locations) to `startLocation`.
        class ReverseShortestPathProblem(SearchProblem):
            def startState(self) -> State:
                """
                Return special "END" state
                """
                pass
                # ### START CODE HERE ###
                # ### END CODE HERE ###

            def isEnd(self, state: State) -> bool:
                """
                Return False for each state.
                Because there is *not* a valid end state (`isEnd` always returns False), 
                UCS will exhaustively compute costs to *all* other states.
                """
                pass
                # ### START CODE HERE ###
                # ### END CODE HERE ###

            def successorsAndCosts(
                self, state: State
            ) -> List[Tuple[str, State, float]]:
                # If current location is the special "END" state, 
                # return all the locations with the desired endTag and cost 0 
                # (i.e, we connect the special location "END" with cost 0 to all locations with endTag)
                # Else, return all the successors of current location and their corresponding distances according to the cityMap
                pass
                # ### START CODE HERE ###
                # ### END CODE HERE ###

        # Call UCS.solve on our `ReverseShortestPathProblem` instance. Because there is
        # *not* a valid end state (`isEnd` always returns False), will exhaustively
        # compute costs to *all* other states.
        # ### START CODE HERE ###
        # ### END CODE HERE ###

        # Now that we've exhaustively computed costs from any valid "end" location
        # (any location with `endTag`), we can retrieve `ucs.pastCosts`; this stores
        # the minimum cost path to each state in our state space.
        #   > Note that we're making a critical assumption here: costs are symmetric!
        # ### START CODE HERE ###
        # ### END CODE HERE ###

    def evaluate(self, state: State) -> float:
        pass
        # ### START CODE HERE ###
        # ### END CODE HERE ###
