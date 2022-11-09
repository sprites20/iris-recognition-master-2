from pathlib import Path    
import json
import glob

path = "./**"
globity = glob.glob(path, recursive=True)

currlowest = 1
currlowestmotionblur = 1
currlowestmix = 1

currhighest = 0
currhighestmotionblur = 0
currhighestmix = 0

motionblurs = 0
normals = 0
mix = 0

motionblursmean = 0
normalsmean = 0
mixmean = 0

for path in globity:
    if path.endswith('json'):
        currjson = json.load(open(path))
        #print(path)
        eyebeingrecognized = currjson["eyebeingrecognized"].split('_')
        groundtruth = currjson["groundtruth"].split('_')
        matchvalue = currjson["best_match_value"]
        if eyebeingrecognized[0] == 'MotionBlur':
            if matchvalue < currlowestmotionblur:
                currlowestmotionblur = matchvalue
            if matchvalue > currhighestmotionblur:
                currhighestmotionblur = matchvalue
            motionblurs = motionblurs + 1
            motionblursmean += matchvalue
        elif eyebeingrecognized[0] == eyebeingrecognized[-3]:
            if matchvalue < currlowest:
                currlowest = matchvalue
            if matchvalue > currhighestmotionblur:
                currhighest = matchvalue
            normals = normals + 1
            normalsmean += matchvalue
        elif eyebeingrecognized[0] == 'motion':
            if matchvalue < currlowestmotionblur:
                currlowestmix = matchvalue
            if matchvalue > currhighestmotionblur:
                currhighestmix = matchvalue
            mix = mix + 1
            mixmean += matchvalue

try:
    motionblursmean /= motionblurs
    normalsmean /= normals
    #mixmean /= mix
except:
    pass
print("LTPS Normals:", currlowest)
print("LTPS Motion:", currlowestmotionblur)

print("HTPS Normals:", currhighest)
print("HTPS Motion:", currhighestmotionblur)

print("Normals Count:", normals)
print("Motions Count:", motionblurs)

print("Normals Mean:",normalsmean)
print("Motions Mean:",motionblursmean)

        #print(json.load(open(path))["eyebeingrecognized"])
        