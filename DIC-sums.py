import os
from pathlib import Path

outputFile = ""

NUM_ATTRIBUTES = 12

def sums():
    # Initialise sum list
    sumCounts = []
    for i in range(NUM_ATTRIBUTES):
        sumCounts.append(0.0)

    line = 0
    with open(outputFile) as aF:
        for outLine in aF:
            out = [float(x) for x in outLine.split(",")]
            attCutOff = len(out) // NUM_ATTRIBUTES
            index = -1
            for i in range(len(out)):
                if i % attCutOff == 0:
                    index = index + 1
                sumCounts[index] = sumCounts[index] + out[i]
            line = line + 1
    print("Sums:\n")
    for j in sumCounts:
        print(j)


def writeDifference(matLine, pyLine):
    mat = [float(x) for x in matLine.split(",")]
    py = [float(x) for x in pyLine.split(",")]

    compare = []
    f = open(outputFile, "a+")
    for index in range(len(mat)):
        f.write(str(abs(mat[index] - py[index])))
        if index != len(mat) - 1:
            f.write(",")
    f.write("\n")
    f.close()


# Reads two files at the same time
def readFile(matFile, pyFile):
    with open(matFile) as mat:
        with open(pyFile) as py:
            for matLine, pyLine in zip(mat, py):
                writeDifference(matLine, pyLine)


def main():

    matFile = input("Enter matlab results file name.")
    pyFile = input("Enter python results file name.")
    global outputFile
    outputFile = input("Enter output file name.")

    # As to not overwrite existing files
    if os.path.isfile(outputFile):
        print("Output file already exists. Try a different name.")
        exit()

    # Create output file
    Path(outputFile).touch()

    # Read in two files and create the output file
    readFile(matFile, pyFile)

    # Print out sums of each attribute
    sums()

main()
