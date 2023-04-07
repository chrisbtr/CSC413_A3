import os
import pickle
from pretty_midi import PrettyMIDI, Instrument, Note, instrument_name_to_program
import numpy as np

import torch
from torch.utils.data import random_split

DATA_DIR = "./data"
TEST_FILE = DATA_DIR + "/albeniz/alb_esp1.mid"
# TEST_FILE = DATA_DIR + "/bach/bach_846.mid"


def parse_directory(dir_name: str, split_size: int = 64):
  print(dir_name)

  # The list of songs that are split into segment sizes
  sequences = []

  # The sum of the times of each song
  total_time = 0

  # the number of songs in the data
  song_count = 0

  for filename in os.listdir(dir_name):
    # If the filename ends with `.mid` or `.MID` parse the file
    # Else it must be a subdirectory so we recursively call `parse_directory`
    # with the subdirectory name
    if filename.endswith(".mid") or filename.endswith(".MID"):
      fname = dir_name + '/' + filename

      # Parse the song file to get the sequences for the song 
      seq, _, time = parse_music_file(fname, split_size=split_size)

      # Append the sequences of the song to the `sequences` list
      sequences.append(seq)

      # Update `total_time` and `song_count`
      total_time += time
      song_count += 1
    else:
      # Recursively get the sequences of the songs in the sub directory
      sub_seqs, times, songs = parse_directory(dir_name + '/' + filename, split_size=split_size)

      # Concat the sub directory sequences the `sequences` list
      sequences += sub_seqs

      # Update `total_time` and `song_count`
      total_time += times
      song_count += songs

  
  return sequences, total_time, song_count

def parse_music_file(fname: str, split_size: int = 64):
  # Initialize a MIDI file object with data from a midi file 
  midi = PrettyMIDI(fname)

  # Use notes being played by the "Piano right" instrument 
  instrument = midi.instruments[0]

  # Sort the notes based on start time
  instrument_notes = sorted(instrument.notes, key=lambda note: note.start)
  
  prev_start = instrument_notes[0].start

  # A list of segments of notes based off `split_size``
  segmented_notes = []

  # The current segment of notes
  curr_split = []

  # A list of notes in the song 
  all_notes = []

  # The total time of the song
  time = midi.get_end_time()

  for note in instrument_notes:
    start = note.start
    end = note.end

    # Compute pitch, step, and duration based on the midi note
    pitch = note.pitch
    step = start - prev_start
    duration = end - start

    curr_split.append([pitch, step, duration])
    all_notes.append([pitch, step, duration])

    prev_start = start

    # Add `curr_split` to the segmented notes
    if len(curr_split) == split_size:
      segmented_notes.append(curr_split)
      curr_split = []

  return segmented_notes, all_notes, time

def create_midi_file(fname: str, notes: list[list[int]]):

  # Create a new MIDI file object
  midi = PrettyMIDI()

  # Create the instrument to be played in the MIDI file object
  instrument = Instrument(program=instrument_name_to_program('Acoustic Grand Piano'))
  
  prev_start = 0
  for note in notes:
    pitch, step, duration = note

    # Create the midi note using the pitch, step, and duration
    start = float(prev_start + step)
    end = float(start + duration)

    midi_note = Note(
      velocity=100,
      pitch=int(pitch),
      start=start,
      end=end,
    )

    # Add the midi note to be played by the instrument
    instrument.notes.append(midi_note)

    prev_start = start

  # Add the instrument to the MIDI file object 
  midi.instruments.append(instrument)

  # Save the midi file 
  midi.write(fname)


if __name__ == '__main__':
  sequences, total_time, total_songs = parse_directory("./data/", split_size=64)

  # Use random_split with a seed to split the songs into training, validation, and testing    
  generator = torch.Generator().manual_seed(42)
  train, validation, test = random_split(sequences, [0.6, 0.2, 0.2], generator=generator)

  # Flatten the training, validation, and testing and convert them into numpy arrays
  train = np.array(sum(train, []))
  validation = np.array(sum(validation, []))
  test = np.array(sum(test, []))

  # Save the dataset as a pickle file
  split_data = train, validation, test
  with open('data.pickle', 'wb') as f:
    pickle.dump(split_data, f)
