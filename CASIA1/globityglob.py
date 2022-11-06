import glob

path = "./**"
globity = glob.glob(path, recursive=True)
for path in globity:
    if path.endswith('jpg'):
        print(path.split('\\')[-1].split('_')[-3], path.split('\\')[-1].split('_')[-2])
    