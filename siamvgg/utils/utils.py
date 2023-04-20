
def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (
                input_length+2*padding[i]-filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width)*get_output_length(height)
