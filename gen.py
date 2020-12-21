import sys
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from src.midi import generate_rand_piano_roll, piano_roll_to_pretty_midi
from src.rnn_models import ConvMidiRNN


if __name__ == '__main__':
    frequency = 4
    notes_num = 88
    first_ch = 21
    last_ch = 108
    seq_length_train = 100
    model_path = 'models/ConvMidiRNN.pth'
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    gen_len = int(sys.argv[3])
    
    im = Image.open(input_path)
    im.load()
    im = im.convert('L')
    # we will use CPU for generation
    input = ToTensor()(im)
    model = ConvMidiRNN(seq_length_train, notes_num)
    # during train I made a mistake by saving models in FastAI with opt fuction an some other params
    # we need only the pytorch model itself
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    
    roll_part = 64 * generate_rand_piano_roll(model, seq_length_train, input, timesteps = gen_len, thresh_r = 0, remove_nan = False)
    # add zero 
    top = torch.zeros((21, roll_part.shape[1]))
    bottom = torch.zeros((128 - 108, roll_part.shape[1]))
    
    roll = torch.cat([top, roll_part, bottom])
    midi = piano_roll_to_pretty_midi(roll, fs = frequency)
    
    midi.write(output_path)
