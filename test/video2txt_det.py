def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # Write code here
    for i in range(len(image)):
        for j in range(len(image[i])):
            r, g, b = image[i][j]
            grayscale_value = 0.299 * r + 0.587 * g + 0.114 * b
            image[i][j] = grayscale_value
    return image


if __name__ == "__main__":
    image = [[[255,0,0]]]
    print(color_to_grayscale(image))