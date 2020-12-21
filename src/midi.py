import pretty_midi
import numpy as np 


def generate_piano_roll(model, seq_length, start_piano_roll, timesteps, thresh = 0.5, remove_nan = False):
  piano_roll = IntToFloatTensor()(ToTensor()(start_piano_roll))
  for i in range(timesteps):
    out = model(piano_roll[None, :, :, -seq_length:]).sigmoid() > thresh
    out = out.unsqueeze(2).byte()
    if remove_nan:
      out = out[:,1:, ]
    piano_roll = torch.cat([piano_roll, out], dim = 2)
  return piano_roll


def generate_rand_piano_roll(model, seq_length, start_piano_roll, timesteps, thresh_r = 0, remove_nan = False):
  piano_roll = IntToFloatTensor()(ToTensor()(start_piano_roll))
  for i in range(timesteps):
    out = model(piano_roll[None, :, :, -seq_length:]).sigmoid()
    r = torch.rand(out.shape)
    out = torch.logical_and(out > r, out > thresh_r)
    out = out.unsqueeze(2).byte()
    if remove_nan:
      out = out[:,1:, ]
    piano_roll = torch.cat([piano_roll, out], dim = 2)
  return piano_roll


# from https://github.com/craffel/pretty-midi/blob/master/examples/reverse_pianoroll.py
def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm
