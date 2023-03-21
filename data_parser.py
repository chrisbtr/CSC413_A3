from mido import MidiFile, second2tick, MetaMessage, Message, MidiTrack, merge_tracks
import torch
import os
import pickle

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
      # try:
      fname = dir_name + '/' + filename
      seq, _, time = parse_music_file(fname)
      sequences.append(seq)
      total_time += time
      total_songs += 1
      # except:
      #   print(f'Warning: issue with {dir_name}/{filename}')
    else:
      sub_seqs, times, songs = parse_directory(dir_name + '/' + filename)
      sequences += sub_seqs
      total_time += times
      total_songs += songs

  
  return sequences, total_time, total_songs

def parse_music_file(fname: str, split_size: int = 64):
  mid = MidiFile(fname, clip=True)
  mid.ticks_per_beat
  notes = []
  curr_split = []
  all_notes = []
  time = 0

  for msg in mid:
    time += msg.time
    # if msg.type == 'note_on' or msg.type == 'note_off':
    if msg.type in ('note_on', 'note_off'):
      # print(msg)
      curr_split.append([msg.note, msg.time, msg.velocity])
      all_notes.append([msg.note, msg.time, msg.velocity])
    else:
      curr_split.append([0, msg.time, 0])
      all_notes.append([0, msg.time, 0])
    if len(curr_split) == split_size:
      notes.append(curr_split)
      curr_split = []
    

  return notes, all_notes, time

def split_midi_by_measure(filename, beats_per_measure):
    # Load MIDI file using mido
    mid = MidiFile(filename)
    mid = merge_tracks(mid.tracks)
    
    # Extract tempo and time signature information
    ticks_per_beat = mid.ticks_per_beat
    tempo = 500000 # default tempo (in case not specified in MIDI file)
    time_signature = (4, 4) # default time signature (in case not specified in MIDI file)
    # for msg in mid:
    #     if msg.type == 'set_tempo':
    #         tempo = msg.tempo
    #     elif msg.type == 'time_signature':
    #         time_signature = (msg.numerator, msg.denominator)
    
    # Calculate ticks per measure based on time signature
    ticks_per_measure = ticks_per_beat * time_signature[0] * 4 / time_signature[1]
    
    # Split MIDI file into measures
    measures = []
    current_measure = []
    ticks = 0
    time = 0
    for msg in mid:
        if msg.type == 'set_tempo':
            # tempo = msg.tempo
            time += msg.time
            current_measure.append([0, msg.time, 0])
        elif msg.type == 'time_signature':
            # time_signature = (msg.numerator, msg.denominator)
            # # time += msg.time
            # ticks_per_measure = ticks_per_beat * time_signature[0] * 4 / time_signature[1]
            pass
        else:#elif msg.type in ('note_on', 'note_off', 'program_change'):
            time += msg.time
            if msg.type in ('note_on', 'note_off'):
              current_measure.append([msg.note, msg.time, msg.velocity])
            else:
              current_measure.append([0, msg.time, 0])
            ticks += second2tick(msg.time, ticks_per_beat, tempo)
            if ticks >= ticks_per_measure * beats_per_measure:
                measures.append(current_measure)
                current_measure = []
                ticks = 0
    print(time)
    return measures

def create_midi_file(fname: str, notes: list[int]):
  # Define some parameters for the MIDI file
  tempo = 600000  # Microseconds per quarter note
  ticks_per_beat = 480

  # Create a new MIDI file object
  mid = MidiFile()

  # Add a new track to the MIDI file
  track = MidiTrack()
  mid.tracks.append(track)

  # Set the tempo and time signature for the track
  track.append(MetaMessage('set_tempo', tempo=tempo))
  track.append(MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24))

  # Iterate over the list of notes and add them to the MIDI track
  for note in notes:
      # Create a new MIDI message for the note
      msg = Message('note_on', note=note[0], velocity=note[2], time=int(note[1]*ticks_per_beat))
      # Add the message to the track
      track.append(msg)
  
  # Save the MIDI file to disk
  mid.save(fname)


if __name__ == '__main__':
  # measures = split_midi_by_measure(TEST_FILE, 4)
  # print(len(measures))
  # print(midi_file)

  # data, all_notes, time = parse_music_file(TEST_FILE)
  # print(time)
  # create_midi_file('new_midi_file.mid', all_notes)
  # for i, notes in enumerate(data):
  #   create_midi_file(f'./tmp/new_midi_file_{i}.mid', notes)
  
  # print(sum([msg[1] for msg in midi_file]))
  sequences, total_time, total_songs = parse_directory("./data/")
  print(len(sequences))
  print(total_songs)
  print(total_time/total_songs)

  generator1 = torch.Generator().manual_seed(42)
  split_data  = torch.utils.data.random_split(sequences, [0.6, 0.2, 0.2], generator=generator1)
  with open('data.pickle', 'wb') as f:
    pickle.dump(split_data, f)
  print(split_data)
