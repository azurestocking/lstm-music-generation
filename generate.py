import numpy as np
import tensorflow as tf
from collections import deque
import midi
import argparse

from constants import *
from util import *
from dataset import *
from tqdm import tqdm
from midi_util import midi_encode



class MusicGeneration:
    """
    encapsulate the logic for maintaining state and generating music based on the model outputs,
    use deque structures for memory of past notes, beats, and style attributes, which helps in providing context for each new generated piece of music
    """

    def __init__(self, style, default_temp=1):
        """
        initialize memories for notes, beats, and styles, and sets up the generation state
        """

        self.notes_memory = deque([np.zeros((NUM_NOTES, NOTE_UNITS)) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        self.beat_memory = deque([np.zeros(NOTES_PER_BAR) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        self.style_memory = deque([style for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)

        # next note being built
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        self.silent_time = NOTES_PER_BAR

        # outputs
        self.results = []

        # temperature
        self.default_temp = default_temp
        self.temperature = default_temp

    def build_time_inputs(self):
        """
        prepare the input set from current memories for the time model
        """

        return (
            np.array(self.notes_memory),
            np.array(self.beat_memory),
            np.array(self.style_memory)
        )

    def build_note_inputs(self, note_features):
        """
        prepare the input set from current memories for the note model
        """

        # timesteps = 1 (no temporal dimension)
        return (
            np.array(note_features),
            np.array([self.next_note]),
            np.array(list(self.style_memory)[-1:])
        )

    def choose(self, prob, n):
        """
        decide on the next note based on the probability distribution provided by the model
        """

        vol = prob[n, -1]
        prob = apply_temperature(prob[n, :-1], self.temperature)

        # flip notes randomly
        if np.random.random() <= prob[0]:
            self.next_note[n, 0] = 1
            # apply volume
            self.next_note[n, 2] = vol
            # flip articulation
            if np.random.random() <= prob[1]:
                self.next_note[n, 1] = 1

    def end_time(self, t):
        """
        finish generation for the current time step, adjust temperature based on activity, and update memories
        """

        # increase temperature while silent
        if np.count_nonzero(self.next_note) == 0:
            self.silent_time += 1
            if self.silent_time >= NOTES_PER_BAR:
                self.temperature += 0.1
        else:
            self.silent_time = 0
            self.temperature = self.default_temp

        self.notes_memory.append(self.next_note)
        # consistent with dataset representation
        self.beat_memory.append(compute_beat(t, NOTES_PER_BAR))
        self.results.append(self.next_note)
        # reset next note
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        return self.results[-1]



def apply_temperature(prob, temperature):
    """
    apply temperature to a sigmoid vector
    """

    # apply temperature
    if temperature != 1:
        # inverse sigmoid
        x = -np.log(1 / prob - 1)
        # apply temperature to sigmoid function
        prob = 1 / (1 + np.exp(-x / temperature))
    return prob

def process_inputs(ins):
    """
    standardize the way inputs are formatted before being fed into the model
    """

    ins = list(zip(*ins))
    ins = [np.array(i) for i in ins]
    return ins

def generate(models, num_bars, styles):
    """
    orchestrate the entire generation process over a specified number of bars, using the models to predict and choose notes sequentially
    """

    print('Generating with styles:', styles)

    # unpack `models` tuple for generating temporal features and notes, respectively
    _, time_model, note_model = models

    # create a list of objects for each style
    generations = [MusicGeneration(style) for style in styles]

    # loop over the number of notes specified by the number of bars
    for t in tqdm(range(NOTES_PER_BAR * num_bars)):
        # for each timestep, each object prepares inputs (note-invariant features) needed for predicting temporal features
        ins = process_inputs([g.build_time_inputs() for g in generations])
        # predict temporal features
        note_features = time_model.predict(ins)
        # extract the features for the last timestep since they are the most relevant for the next note generation
        note_features = np.array(note_features)[:, -1:, :]

        # generate each note conditioned on previous notes
        for n in range(NUM_NOTES):
            # for each timestep, each object prepares inputs needed for predicting the next note based on the last time step's note features and the current state of the generation
            ins = process_inputs([g.build_note_inputs(note_features[i, :, :, :]) for i, g in enumerate(generations)])
            # predict the next note
            predictions = np.array(note_model.predict(ins))
            # choose the next note
            for i, g in enumerate(generations):
                # select the most relevant (latest) prediction output for the current note position, 
                # removing the temporal dimension by focusing only on the current prediction without considering previous temporal steps
                g.choose(predictions[i][-1], n)

        # finalize the generation process for the current timestep and prepare for the next,
        # enabling real-time interactions 
        yield [g.end_time(t) for g in generations]

def write_file(name, results):
    """
    output the generated music to MIDI files:
    take a list of all notes generated per track and writes it to file
    """

    results = zip(*list(results))

    for i, result in enumerate(results):
        fpath = os.path.join(SAMPLES_DIR, name + '_' + str(i) + '.mid')
        print('Writing file', fpath)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        mf = midi_encode(unclamp_midi(result))
        midi.write_midifile(fpath, mf)



def main():
    """
    set up command-line argument parsing and drives the generation based on user inputs
    """

    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--bars', default=32, type=int, help='Number of bars to generate')
    parser.add_argument('--styles', default=None, type=int, nargs='+', help='Styles to mix together')
    args = parser.parse_args()

    models = build_or_load()

    # create a one-hot encoded vector for each style index, then take the mean of all the styles, resulting in a new vector where each element is the average of the corresponding elements in all the input vectors
    # contain no specific logic to handle or remove duplicates before averaging
    if args.styles:
        styles = [np.mean([one_hot(i, NUM_STYLES) for i in args.styles], axis=0)]
    else:
        styles = [compute_genre(i) for i in range(len(genre))]

    write_file('output', generate(models, args.bars, styles))

if __name__ == '__main__':
    main()
