from PIL import Image
import os

# set the dimensions of the final GIF collage
width = 800
height = 600

# set the folder containing the input GIFs
input_folder = "path/to/folder/containing/gifs"

# create an empty list to store the GIF frames
frames = []

# loop through all the GIF files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".gif"):
        # open the GIF file and append its frames to the list
        with Image.open(os.path.join(input_folder, filename)) as im:
            frames.extend(im.seek(i) for i in range(im.n_frames))

# create a new image with the desired dimensions
result = Image.new("RGB", (width, height), (255, 255, 255))

# calculate the dimensions of each GIF in the collage
gif_width = width // len(frames)
gif_height = height // len(frames)

# loop through all the GIF frames and paste them onto the result image
for i, frame in enumerate(frames):
    # resize the frame to fit within the dimensions of each GIF
    frame = frame.resize((gif_width, gif_height))
    # calculate the position of the frame in the collage
    x = i % width
    y = i // width
    # paste the frame onto the result image at the correct position
    result.paste(frame, (x * gif_width, y * gif_height))

# save the result image as a GIF file
result.save("path/to/output/collage.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
