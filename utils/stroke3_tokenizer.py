import numpy as np
import pandas as pd
from sketch import convert_sketch_from_stroke3_to_image
from PIL import Image

def find_min_max(strokes):
    max = -1000

    for i in range(len(strokes)):
      abs_array = np.absolute(strokes[i])
      if np.max(abs_array) > max:
        max = np.max(abs_array)

    return max


def normalize_strokes(strokes):
    """
    Normalize strokes and set their values between a [-1,1] range
    """
    scale = find_min_max(strokes)

    for i in range(len(strokes)):
        strokes[i] /= np.max(scale,axis=0)

    return (scale, strokes)

def get_sketch_coordinates(sketch, scale_factor=1.0, start_from_origin=False):
    """
    This method takes the sketch in stroke3 format and 
    calculates the coordinates of the points which compose
    each stroke. 

    sketch : sketch in stroke3 format
    scale_factor : factor by which the coordinates can be scaled
    start_from_origin : boolean value which determines whether the
                        sketch starts from the origin (0,0) or not

    return : list of strokes, being that each stroke is composed of
             n (x,y) coordinates 
    """
    x_coord = 0
    y_coord = 0
    dx = 0
    dy = 0

    strokes = []
    stroke = [[0, 0]] if start_from_origin else []
    for i in range(len(sketch)):
        dx, dy = sketch[i, :2] * scale_factor
        
        x_coord += dx
        y_coord += dy
        stroke.append([x_coord, y_coord])

        if sketch[i, 2] == 1:
            final_stroke = np.array(stroke) + np.zeros((1, 2), dtype=np.uint8)
            strokes.append(final_stroke)
            stroke = []
    
    return strokes

def coordinates_to_stroke3(coords, omit_first_point=True):
    """
    Convert sketch coordinates into original stroke3 format

    coords: list of coords (x,y)
    """
    strokes = []
    for coord in coords:
        coord_len = len(coord)
        for i in range(coord_len):
            eos = 0 if i < coord_len - 1 else 1
            strokes.append([coord[i][0], coord[i][1], eos])

    strokes = np.array(strokes)
    strokes[1:, 0:2] -= strokes[:-1, 0:2]

    return strokes[1:, :] if omit_first_point else strokes

def stroke3_to_image(sketch, scale, file_name="test.png", image_resolution=1000):
    sketch[:, :2] = sketch[:, :2] / scale

    img_array = convert_sketch_from_stroke3_to_image(sketch, image_resolution) 
    img_array = (img_array * 255.0).astype(np.uint8)
    img_array = np.squeeze(img_array)

    img = Image.fromarray(img_array, 'L')
    img.save(file_name)





class GridTokenizer(object):
    """
    Grid Tokenizer

    The sketch is divided into several grid cells and each cell is associated 
    with a token.
    """

    def __init__(self, resolution=100, max_seq_len=0):
        """
        resolution : defines how many grid cells will represent the sketch.
                     This value is given by resolution^2, which essentially means
                     that the greater the resolution the more detail the tokenized
                     outputs will have regarding the stroke positions
        
        max_seq_len : defines a limit to the size of the stroke sequence
        """
        self.radius = int(resolution / 2)
        self.resolution = self.radius * 2
        self.radius_grid_cell = 1 / self.resolution
        self.max_seq_len = max_seq_len

        # string special tokens used to organize the sketch and its strokes
        self.SEP = self.resolution**2 + 1
        self.SOS = self.SEP + 1
        self.EOS = self.SEP + 2
        self.PAD = 0

        self.VOCAB_SIZE = self.resolution**2 + 4


    def encode(self, sketch, seq_len=0):
      """
        The enconding process works as follows: we take the sketch representation
        in the stroke3 format and convert it into (x,y) coordinates, which should 
        be followed by a normalization so as to convert the values into tokens properly.
        We then multiply each coordinate by the radius so as to convert them
        into the nxn proportion (considering both negative and positive numbers).
        Then theses values are added up, finally representing the tokens associated
        with each coordinate, which form a stroke.

        Note that the special tokens must be used to organize the final output and 
        the padding is used to give the strokes the same size in order to make them
        uniform.

        sketch : drawing in stroke3 format
        seq_len : predefined sequence length

        return : tuple consisting of the scale factor and the tokenized sketch
      """
      strokes = get_sketch_coordinates(sketch)
      norm_scale, strokes = normalize_strokes(strokes)

      tokenized_sketch = []
      for stroke in strokes:
        x_tokens = np.int64((stroke[:, 0] + 1) * self.radius)  # we add 1 to the coordinates E [-1,1] so the token values stay between [0,resolution]
        x_tokens[x_tokens == self.resolution] = self.resolution - 1  # if the resulting value is 100 then we subtract one in order to deal with the upper bound
        y_tokens = np.int64((stroke[:, 1] + 1) * self.radius)
        y_tokens[y_tokens == self.resolution] = self.resolution - 1

        stroke_tokens = (x_tokens * self.resolution) + y_tokens  # by adding x and y values this way we can decode the tokens using the quotient and the remainder
        stroke_tokens = list(stroke_tokens + 1) + [self.SEP]  # the tokens start from 1 in order for us to use 0 as out padding special token
        tokenized_sketch.extend(stroke_tokens)

      # adding beggining and ending special tokens
      tokenized_sketch = [self.SOS] + tokenized_sketch + [self.EOS]

      # here we add padding if necessary
      if self.max_seq_len > 0: 
          units_padding = self.max_seq_len - len(tokenized_sketch)
          if units_padding > 0:
              tokenized_sketch += [self.PAD] * units_padding
          else:
              tokenized_sketch = tokenized_sketch[:self.max_seq_len]
              tokenized_sketch[-2:] = [self.SEP, self.EOS]

      if len(tokenized_sketch) < seq_len:
          tokenized_sketch += [self.PAD] * (seq_len - len(tokenized_sketch))

      return (norm_scale, np.array(tokenized_sketch))

    def decode(self, sketches):
        if len(sketches) > 0 and isinstance(sketches[0], (list, tuple, np.ndarray)):
            return self.decode_list(sketches[0], sketches[1])
        else:
            return self.decode_single(sketches[0], sketches[1])

    def decode_single(self, tokenized_sketch, norm_scale):
        """
        When it comes to the decoding part (tokens -> stroke3) we take 
        the quotient to represent one coordinate value and the module to  
        represent the other and then divide it by the radius so we can 
        convert the value to its original [-1,1] normalized scale.

        tokenized_sketch: tokenized sketch comprised of n tokenized strokes
        
        return : decoded sketch in stroke3 format
        """

        stroke3_sketch = []
        stroke = []
        for token in tokenized_sketch:
            if 0 < token < self.SEP:  # checks if token is not a special token
                x_coords = (token - 1) // self.resolution  # the 1 we added to reserve the 0 for padding is subtracted
                x_coords = ((x_coords / self.radius) - 1 + self.radius_grid_cell) * norm_scale  # we subtract 1 so the values range from [-1,1] instead of [0,1] as was the case in the encoding
                y_coords = (token - 1) % self.resolution  
                y_coords = ((y_coords / self.radius) - 1 + self.radius_grid_cell) * norm_scale 

                stroke.append(np.array([x_coords, y_coords]))
            elif token == self.SEP and stroke:
                stroke3_sketch.append(np.array(stroke))
                stroke = []
            elif token == self.EOS:
                break

        stroke3_sketch = coordinates_to_stroke3(stroke3_sketch, omit_first_point=False)
        return stroke3_sketch




def main():

    stroke3 = []
    with open("normalized_sketch.txt", 'r') as sketch:
        lines = sketch.readlines()
        for line in lines:
            stroke3.append(list(map(float, line.rstrip().split(' '))))

    example = np.array(stroke3)
    print('Original sketch:\n')
    print(example, '\n')

    grid_tok = GridTokenizer(max_seq_len=1000, resolution=10000)      
    scale, sketch = grid_tok.encode(example)
    print('Encoded sketch:\n')
    print(sketch, '\n')

    decoded_sketch = grid_tok.decode_single(sketch, scale)
    print('Decoded sketch:\n')
    print(decoded_sketch, '\n')

    stroke3_to_image(decoded_sketch, scale, file_name='decoded_eiffel_tower_10000_res.png')

if __name__ == "__main__":
    main()

    