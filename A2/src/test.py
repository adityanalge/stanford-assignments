from mapUtil import (
    CityMap,
    checkValid,
    createGridMap,
    createGridMapWithCustomTags,
    createStanfordMap,
    getTotalCost,
    locationFromTag,
    makeGridLabel,
    makeTag,
)

import grader
import util
import submission

def printPath(
    path,
    waypointTags,
    cityMap,
    outPath = "path.json",
):
    doneWaypointTags = set()
    for location in path:
        for tag in cityMap.tags[location]:
            if tag in waypointTags:
                doneWaypointTags.add(tag)
        tagsStr = " ".join(cityMap.tags[location])
        doneTagsStr = " ".join(sorted(doneWaypointTags))
        print(f"Location {location} tags:[{tagsStr}]; done:[{doneTagsStr}]")
    print(f"Total distance: {getTotalCost(path, cityMap)}")

    # (Optional) Write path to file, for use with `visualize.py`
    if outPath is not None:
        with open(outPath, "w") as f:
            data = {"waypointTags": waypointTags, "path": path}
            json.dump(data, f, indent=2)

cityMap=createGridMap(3, 5)
startLocation=makeGridLabel(0, 0)
endTag=makeTag("label", makeGridLabel(2, 2))
expectedCost=4

ucs = util.UniformCostSearch(verbose=5)
ucs.solve(submission.ShortestPathProblem(startLocation, endTag, cityMap))

path = grader.extractPath(startLocation, ucs)

if (checkValid(path, cityMap, startLocation, endTag, [])):
    print("Path is Valid")
if expectedCost is not None and expectedCost == getTotalCost(path, cityMap):
    print("True")

ucs.solve(submission.getStanfordShortestPathProblem())
path = extractPath(problem.startLocation, ucs)
printPath(path=path, waypointTags=[], cityMap=stanfordMap)

self.assertTrue(checkValid(path, stanfordMap, problem.startLocation, problem.endTag, []))

self.skipTest("This test case is a helper function for students.")

