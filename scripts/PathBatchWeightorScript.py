import os;
import sys;
import subprocess;

print("Running: " , sys.argv[0])
print("Arguments: ", str(sys.argv))

numRuns = int(sys.argv[1])
numPaths = int(sys.argv[2])
outputDirectory = sys.argv[3]

exeToRun = "./build/Utils/Release/PathBatchWeightor.exe"

# Create necessary directory
if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)

currentExperimentDir = outputDirectory + "/" + str(numPaths)
if not os.path.exists(currentExperimentDir):
    os.makedirs(currentExperimentDir)

print (os.getcwd())

for expNum in range(0, numRuns):
    pathBatchFile = currentExperimentDir + "/PathBatchGen_" + str(numPaths) + "_" + str(expNum) + "_Path_Output.tpb"
    arguments = [exeToRun, pathBatchFile, str(200)]
    subprocess.call(arguments)