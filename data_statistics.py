import pickle
import matplotlib.pyplot as plt

path_to_data = './data.pickle'

# NOTE: Un comment the below lines if running on colab

# from google.colab import drive

# # Mount drive
# drive.mount('/content/gdrive')
# # UPDATE your path to the data
# path_to_data = '/content/gdrive/My Drive/University/Year 4/CSC413/Project/data.pickle'

# Load the dataset
with open(path_to_data, 'rb') as f:
  dataset = pickle.load(f)

train_set, validation_set, test_set = dataset


print(train_set.shape)
print(validation_set.shape)
print(test_set.shape)

# Generate histograms of the notes for each data set.
train_pitch_histo = []
valid_pitch_histo = []
test_pitch_histo = [] 

pitch_histos = [train_pitch_histo, valid_pitch_histo, test_pitch_histo]

# discretize step and duration into bins 
train_step_histo = [] 
valid_step_histo = [] 
test_step_histo = [] 

step_histos = [train_step_histo, valid_step_histo, test_step_histo]

train_dur_histo = []
valid_dur_histo = [] 
test_dur_histo = [] 

dur_histos = [train_dur_histo, valid_dur_histo, test_dur_histo]

set_names = ["Training", "Validation", "Testing"]
note_names = ["Pitch", "Step", "Duration"]
all_histos = [pitch_histos, step_histos, dur_histos]

for i, data_set in enumerate([train_set, validation_set, test_set]):
  for sequence in train_set:
    for note in sequence:
      pitch, step, dur = note
      pitch_histos[i].append(int(pitch))
      step_histos[i].append(step)
      dur_histos[i].append(dur)
  
  fig, ax = plt.subplots(1, len(note_names))
  for j, distro in enumerate(note_names):
    ax[j].set_title(f"{set_names[i]} {distro} Distribution")
    ax[j].hist(all_histos[j][i])
    
  fig.tight_layout(pad=0.5)
  plt.show()