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
    model_path = 'models/ConvMidiRNN.pth'
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    gen_len = int(sys.argv[2])
    
    im = Image.open(fn)
    im.load()
    im = im.convert('L')
    # we will use CPU for generation
    input = ToTensor()(im).unsqueeze(0)
    model = ConvMidiRNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    roll_part = 64 * generate_rand_piano_roll(input, timesteps = gen_len, thresh_r = 0, remove_nan = False)
    # add zero 
    top = torch.zeros((21, roll_part.shape[1]))
    bottom = torch.zeros((128 - 108, roll_part.shape[1]))
    
    roll = torch.cat([top, roll_part, bottom])
    midi = piano_roll_to_pretty_midi(roll, fs = frequency)
    
    midi.write(output_path)
