Before training data can be generated, first background image dataset need to be prepared via

1) temporal median filtering -> 1 bg image per video sequence
2) BG algorithm -> use bgslibrary and store 1 bg image per video frame

bg image dataset needs to have the same folder structure as the CDnet2014:
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


1) If 1 bg image is used per video sequence -> place rgb background image in the respective folders
 and name it "background.jpg"

2) If 1 bg image per video frame is generated -> place as many bg images in the folders as the respective video sequence
	and name the bg images according to this format -> "bin000001.png" - bin00XXXX.png"

NOTE: If SubSense's bg model is used the stored background images have a black boundary -> remove it with nearby pixels (see padBg.m)or otherwise
	  the bg image is corrupted at the boundary
