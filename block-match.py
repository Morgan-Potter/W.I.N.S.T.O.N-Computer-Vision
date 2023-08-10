

# def ssd(block1: list, block2: list) -> int:
#     """ Returns the sum of squared differences between two blocks """
#     out = 0
#     size = len(block1)
#     if size != len(block2):
#         raise Exception("ERROR: Blocks are not the same size.")
#     for i in range(size):
#         out += (block1[i] - block2[i])**2
#     return out

# def match_images(img1, img2, block_size: int) -> list:
#     """ ## Returns a list of matching pixels formatted [i_1(x,y), i_2(x,y)] """
#     img1y = range(len(img1[0]))
#     img1x = range(len(img1))
#     img2x = range(len(img2))
#     block_max = int(block_size/2)
#     block_x = range(-block_max, block_max + 1)
#     block_y = range(block_max, -(block_max+1), -1)
#     out = []
#     for j in img1y:
#         for i in img1x:
#             costs = []
#             for x in img2x:
#                 total = 0
#                 for wy in block_y:
#                     for wx in block_x:
#                         total += (img1[j + wy][i + wx] - img1[j + wy][x + wx])**2
#                 costs.append(total)
#         out.append(costs)
#         print(costs)
#     return out
class image_match():
    def __init__(self, img1, img2, block_size, threshold):
        self.img1 = img1
        self.img2 = img2
        self.block_size = block_size
        self.threshold = threshold
        self.average = 0
    
    def _map(self, val, from_min, from_max, to_min, to_max) -> int:
        return round((val - from_min) / (from_max - from_min) * (to_max - to_min) + to_min)

    def threshold_match(self, threshold):
        img1 = self.img1
        img2 = self.img2
        threshold = self.threshold
        block_max = int(self.block_size/2)
        img1x = range(block_max, len(img1[0]) - block_max)
        img1y = range(block_max, len(img1) - block_max)
        block = range(-block_max, block_max+1)
        img2max = len(img2[0]) - block_max
        img2x = None
        out = []
        for j in img1y:
            row = []
            for i in img1x:
                img2i = block_max
                lowest_tot = float('inf')
                direction = True
                img2x = i
                while img2max > img2x: # /2 needed as it iterates once every 2 cycles
                    total = 0
                    # Determine x coordinate by flipping iterator direction
                    if direction:
                        img2x = i + img2i
                    else:
                        img2x = i - img2i
                    if img2x-block_max < 0:
                        direction = not direction
                        continue
                    if img2x >= img2max:
                        direction = not direction
                        continue

                    # Determine SSD of block
                    for wy in block:
                        for wx in block:
                            total += abs(int(img1[j + wy][i + wx]) - int(img2[j + wy][img2x + wx]))
                    
                    # Determine current lowest value
                    if total < lowest_tot:
                        lowest = img2x
                        lowest_tot = total
                    
                    # Terminate if the total is below the match threshold
                    if lowest_tot < threshold:
                        break
                    img2i += direction # Iterate once every two cycles
                    direction = not direction # Change direction
                row.append(self._map((i-lowest), -len(img1[0]), len(img1[0]), 0, 255))
            print('row', j, 'complete')
            out.append(row)
        return out
    def threshold_match_row(self, row):
        threshold = self.threshold
        img1 = self.img1
        img2 = self.img2
        block_max = int(self.block_size/2)
        img1x = range(block_max, len(img1[0]) - block_max)
        block = range(-block_max, block_max+1)
        img2max = len(img2[0]) - block_max
        img2x = None
        out = []
        for i in img1x:
            img2i = block_max
            lowest_tot = float('inf')
            direction = True
            img2x = i
            while img2max > img2x: # /2 needed as it iterates once every 2 cycles
                total = 0
                # Determine x coordinate by flipping iterator direction
                if direction:
                    img2x = i + img2i
                else:
                    img2x = i - img2i
                if img2x-block_max < 0:
                    direction = not direction
                    continue
                if img2x >= img2max:
                    direction = not direction
                    continue

                # Determine SSD of block
                for wy in block:
                    for wx in block:
                        total += (int(img1[row + wy][i + wx]) - int(img2[row + wy][img2x + wx]))**2
                
                # Determine current lowest value
                if total < lowest_tot:
                    lowest = img2x
                    lowest_tot = total
                
                # Terminate if the total is below the match threshold
                if lowest_tot < threshold:
                    break
                img2i += direction # Iterate once every two cycles
                direction = not direction # Change direction
            out.append(self._map(lowest, 0, img2max, 0, 255))
        print('row', row, 'complete')

        return out
    
    def match_image_row(self, row: int) -> list:
        """ ## Returns a list of matching pixels formatted [i_1(x,y), i_2(x,y)] """
        img1 = self.img1
        img2 = self.img2
        block_size = self.block_size
        img1x = range(len(img1[0]) - 2)
        img2x = range(len(img2[0]) - 2)
        block_max = int(block_size/2)
        block_x = range(-block_max, block_max + 1)
        block_y = range(block_max, -(block_max+1), -1)
        out = []
        for i in img1x:
            lowest_tot = float('inf')
            for x in img2x:
                total = 0
                for wy in block_y:
                    for wx in block_x:
                        total += (int(img1[row + wy][i + wx]) - int(img2[row + wy][x + wx]))**2
                if lowest_tot > total:
                    lowest = x
                    lowest_tot = total + 1
            out.append(self._map((i-lowest), -len(img1[0]), len(img1[0]), 0, 255))
        print('row', row, 'complete')
        return out
def average_brightness(img):
    total = 0
    count = 0
    for y in img:
        for x in y:
            total += x
            count += 1
    return total / count
if __name__ == '__main__':
    import cv2
    import numpy
    import json
    import math
    def standard_deviation(img, average):
        total_deviation = 0
        count = 0
        for y in img:
            for x in y:
                total_deviation += (x - average)**2
                count += 1
        return math.sqrt(total_deviation/count)
    from multiprocessing import Pool
    img1 = cv2.cvtColor(cv2.imread('./calibration-images/left/13.png'), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.imread('./calibration-images/right/13.png'), cv2.COLOR_BGR2GRAY)
    p = Pool(processes=61)
    result = p.map(image_match(img1, img2, 7, 205).threshold_match_row, [i for i in range(2, len(img1) - 2)])
    open('test_disparity7.json', 'w').write(json.dumps(result))
    a = numpy.array(json.load(open('test_disparity7.json')), numpy.uint8)
    # average = average_brightness(a)
    # stdev = standard_deviation(a, average)
    # print(average + stdev)
    # print(average_brightness(a))
    # a1 = numpy.array(json.load(open('test_disparity5.json')), numpy.uint8)
    # a2 = numpy.array(json.load(open('test_disparity3.json')), numpy.uint8)
    # cv2.imwrite('disparity-map.png', a)
    # cv2.imshow('disparity-map-100000', a)
    # cv2.imshow('disparity-map-10000', a1)
    cv2.imshow('disparity-map', a)
    # cv2.imshow('disparity-map-no-equivalent-matches', a2)
    cv2.waitKey(0)




                
