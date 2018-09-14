2 possibilities to generate the training data
Background images should be prepared beforehand!

1) temporal median bg images -> 1 bg image per video scene
2) bg model bg images -> 1 bg image per video frame

-> provide background root path: it should have following structure

bg_root
├── badWeather
│   ├── blizzard
│   ├── skating
│   ├── snowFall
│   └── wetSnow
├── baseline
│   ├── highway
│   ├── office
│   ├── pedestrians
│   └── PETS2006
├── cameraJitter
│   ├── badminton
│   ├── boulevard
│   ├── sidewalk
│   └── traffic
├── dynamicBackground
│   ├── boats
│   ├── canoe
│   ├── fall
│   ├── fountain01
│   ├── fountain02
│   └── overpass
├── intermittentObjectMotion
│   ├── abandonedBox
│   ├── parking
│   ├── sofa
│   ├── streetLight
│   ├── tramstop
│   └── winterDriveway
├── lowFramerate
│   ├── port_0_17fps
│   ├── tramCrossroad_1fps
│   ├── tunnelExit_0_35fps
│   └── turnpike_0_5fps
├── nightVideos
│   ├── bridgeEntry
│   ├── busyBoulvard
│   ├── fluidHighway
│   ├── streetCornerAtNight
│   ├── tramStation
│   └── winterStreet
├── PTZ
│   ├── continuousPan
│   ├── intermittentPan
│   ├── twoPositionPTZCam
│   └── zoomInZoomOut
├── shadow
│   ├── backdoor
│   ├── bungalows
│   ├── busStation
│   ├── copyMachine
│   ├── cubicle
│   └── peopleInShade
├── thermal
│   ├── corridor
│   ├── diningRoom
│   ├── lakeSide
│   ├── library
│   └── park
└── turbulence
    ├── turbulence0
    ├── turbulence1
    ├── turbulence2
    └── turbulence3


for 1) use PatchCreatorBlock.py and adapt the absolute paths respectively
	2) use SubSensePatchCreatorBlock.py and adapt the absolute paths respectively

IMPORTANT: 1) Prepare ROI.png instead of ROI.bmp for each video sequence in dataset 2014
				-> best pratice: open ROI.bmp with default program -> "save as" -> "ROI.png" 
		   2) before runing the script, provide output folder with following structure
			-> prepare three folders within output root

├── out_root
    ├── stats
    ├── train
    └── val

After script is done: - run the matlab file getStatsHdf5.m and adapt the paths respectively
					  - set the output root to the stats folder within the out_root folder
					  - variables "start" and "end_id" are given in the train folder of the outputroot

e.g. in "train" folder : data000000.h5 - data000010.h5
start = 0,end_id = 10

After the matlab script, everything should be set up for the training

Test: In order to test the correct preparation, after adapting the paths in scripts/show_minibatch.lua,
	  run the file 
	  
IMPORTANT: run with "qlua" instead of "th"



