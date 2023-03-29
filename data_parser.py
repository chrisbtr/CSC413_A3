import torch
import os
import pickle
import pretty_midi
import numpy as np

DATA_DIR = "./data"
TEST_FILE = DATA_DIR + "/albeniz/alb_esp1.mid"
# TEST_FILE = DATA_DIR + "/bach/bach_846.mid"


def parse_directory(dir_name: str):
  print(dir_name)
  sequences = []
  total_time = 0
  total_songs = 0
  for filename in os.listdir(dir_name):
    if filename.endswith(".mid") or filename.endswith(".MID"):
      fname = dir_name + '/' + filename
      seq, _, time = parse_music_file(fname)
      sequences.append(seq)
      total_time += time
      total_songs += 1
    else:
      sub_seqs, times, songs = parse_directory(dir_name + '/' + filename)
      sequences += sub_seqs
      total_time += times
      total_songs += songs

  
  return sequences, total_time, total_songs

def parse_music_file(fname: str, split_size: int = 64):
  midi = pretty_midi.PrettyMIDI(fname)
  instrument = midi.instruments[0]
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  notes = []
  curr_split = []
  all_notes = []
  time = 0

  for note in sorted_notes:
    start = note.start
    end = note.end

    curr_split.append([note.pitch, start - prev_start, end - start])
    all_notes.append([note.pitch, start - prev_start, end - start])
    prev_start = start

    if len(curr_split) == split_size:
      notes.append(curr_split)
      curr_split = []

  return notes, all_notes, time

def create_midi_file(fname: str, notes: list[int]):

  # Create a new MIDI file object
  midi = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
  
  prev_start = 0
  for note in notes:
    start = float(prev_start + note[1])
    end = float(start + note[2])
    note = pretty_midi.Note(
        velocity=100,
        pitch=int(note[0]),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  midi.instruments.append(instrument)
  midi.write(fname)


if __name__ == '__main__':
  sequences, total_time, total_songs = parse_directory("./data/")

  generator1 = torch.Generator().manual_seed(42)
  train, validation, test = torch.utils.data.random_split(sequences, [0.6, 0.2, 0.2], generator=generator1)

  train = np.array(sum(train, []))
  validation = np.array(sum(validation, []))
  test = np.array(sum(test, []))

  split_data = train, validation, test

  with open('data.pickle', 'wb') as f:
    pickle.dump(split_data, f)
