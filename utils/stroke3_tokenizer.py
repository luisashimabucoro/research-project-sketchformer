import numpy as np
import pandas as pd

class GridTokenizer(object):

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
        self.PAD = -2

        self.VOCAB_SIZE = self.resolution**2 + 4

    def strokes_to_lines(self, strokes, scale=1.0, start_from_origin=False):
        """
        convert strokes3 to polyline format ie. absolute x-y coordinates
        note: the sketch can be negative
        :param strokes: stroke3, Nx3
        :param scale: scale factor applied on stroke3
        :param start_from_origin: sketch starts from [0,0] if True
        :return: list of strokes, each stroke has format Nx2
        """
        x = 0
        y = 0
        lines = []
        line = [[0, 0]] if start_from_origin else []
        for i in range(len(strokes)):
            x_, y_ = strokes[i, :2] * scale
            x += x_
            y += y_
            line.append([x, y])
            if strokes[i, 2] == 1:
                line_array = np.array(line) + np.zeros((1, 2), dtype=np.uint8)
                lines.append(line_array)
                line = []
        if lines == []:
            line_array = np.array(line) + np.zeros((1, 2), dtype=np.uint8)
            lines.append(line_array)
        return lines

    # Takes strokes from a sketch and converts them to tokens
    # 1 - 
    def encode(self, sketch, seq_len=0):
        r = self.strokes_to_lines(sketch, 1.0)
        print(r)

        # stroke3s -> tokens
        out = []
        for stroke in r:
            print(stroke[:, 0])
            x_t = np.int64((stroke[:, 0] + 1) * self.radius)
            x_t[x_t == self.resolution] = self.resolution - 1  # deal with upper bound
            print(x_t)
            y_t = np.int64((stroke[:, 1] + 1) * self.radius)
            y_t[y_t == self.resolution] = self.resolution - 1
            print(y_t)
            t_id = x_t + y_t * self.resolution
            t_id = list(t_id + 1) + [self.SEP]  # shift by 1 to reserve id 0 for PAD
            print(t_id)
            out.extend(t_id)
        out = [self.SOS] + out + [self.EOS]
        if self.max_seq_len:  # pad
            npad = self.max_seq_len - len(out)
            if npad > 0:
                out += [self.PAD] * npad
            else:
                out = out[:self.max_seq_len]
                out[-2:] = [self.SEP, self.EOS]
        if len(out) < seq_len:
            out += [self.PAD] * (seq_len-len(out))
        return np.array(out)


def main():

    stroke3 = []
    with open("normalized_sketch.txt", 'r') as sketch:
        lines = sketch.readlines()
        for line in lines:
            stroke3.append(list(map(float, line.rstrip().split(' '))))
    
    example = np.array(stroke3)
    # print(example)
    # print(np.int64(example[:, 0]))

    tok = GridTokenizer()      
    print(tok.encode(example, seq_len=0))


if __name__ == "__main__":
    main()