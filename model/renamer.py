import os
import sys

counter = 1

path = "./9"

commandLineArgs = sys.argv

if len(commandLineArgs) > 1:
  pattern = commandLineArgs[1] + "_{}"
else:
  print('Enter a pattern for the new filenames as a command line argument')
  sys.exit()

for filename in os.listdir(path):
  print("Renaming: " + filename + "...")
  file_ext = os.path.splitext(filename)[1] 
  new_filename = pattern.format(counter) + file_ext
  oldFileName = os.path.join(path, filename)
  newFileName = os.path.join(path, new_filename)
  os.rename(oldFileName, newFileName)
  counter += 1
  
print("All files renamed.")