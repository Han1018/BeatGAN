from pypianoroll import Multitrack
import numpy as np
import glob2 as gb

# Parameters
FILENAMES = ['./genre_classify/blues/0d15ccb09ec9efa67f933ec37c56c5eb.mid'] # The input MIDI filenames
folderPath='./new_classify/rock_pop/'
RESULT_FILENAME= './train.npz' # The resulting filename
N_TRACKS = 5
BEAT_RESOLUTION = 12 # The beat resolution

# Initialize an empty list to collect the results
results = []

# Pass times
passTimes = 0

# total songs
transTimes = 0


#print('hi',gb.glob(folderPath+'*.mid'))

# Iterate through all the MIDI files
#for filename in FILENAMES:
for filename in gb.glob(folderPath+'*.mid'):
    # Parse the MIDI file into multitrack pianoroll
    try:
        multitrack  = Multitrack(filename, beat_resolution=BEAT_RESOLUTION)
    except:
        continue

    # Pad to multtple
    multitrack.pad_to_multiple(4 * BEAT_RESOLUTION)

    # Binarize the pianoroll
    multitrack.binarize()

    # Sort the tracks according to program number
    multitrack.tracks.sort(key=lambda x: x.program)

    # Bring the drum track to the first track
    multitrack.tracks.sort(key=lambda x: ~x.is_drum)

    try:
        # Get the stacked pianoroll
        pianoroll = multitrack.get_stacked_pianorolls()
    except:
        continue

    # Check length
    if pianoroll.shape[0] < 4 * 4 * BEAT_RESOLUTION:
        continue
    
    try:
        # Keep only the mid-range pitches
        pianoroll = pianoroll[:, 24:108]
    except:
        continue

    try:
        # Reshape and get the phrase pianorolls
        # origin
        #pianoroll = pianoroll.reshape(-1, 4 * BEAT_RESOLUTION, 84, N_TRACKS)
        # changed by paul

        pianoroll = pianoroll.reshape(-1,4, 4*BEAT_RESOLUTION, 84, N_TRACKS)
        #print(pianoroll.shape)
        #results.append(np.concatenate(
        #    [pianoroll[:-3], pianoroll[1:-2], pianoroll[2:-1], pianoroll[3:]], 1))
        results.append(pianoroll)
        transTimes = transTimes + 1
    except:
        continue
result = np.concatenate(results, 0)
print('times',': ',transTimes,'shape:',result.shape)
# NOTE: You might want to shuffle the training data here
np.savez_compressed(
    RESULT_FILENAME, nonzero=np.array(result.nonzero()),
    shape=result.shape)
