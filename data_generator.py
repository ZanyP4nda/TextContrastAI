from random import randint
import pickle

dataset_filename = 'train_dataset.pkl'

white_fg_relative_luminance = 1.05
black_fg_relative_luminance = 0.05

def dec_to_hex(n):
	return hex(n)[2:]

def rgb_to_hex(rgb):
	return "".join([dec_to_hex(x) for x in rgb])

def get_rand_bg_color():
	return [randint(0, 255), randint(0, 255), randint(0, 255)]

def get_optimum_fg_color(bg_rgb):
	bg_srgb = [x/255 for x in bg_rgb]

	bg_linear_rgb = []
	for i in range(len(bg_srgb)):
		if bg_srgb[i] > 0.03928:
			bg_linear_rgb.append(((bg_srgb[i]+0.055)/1.055)**2.4)
		elif bg_srgb[i] <= 0.03928:
			bg_linear_rgb.append(bg_srgb[i] / 12.92)

	bg_relative_luminance = bg_linear_rgb[0] * 0.2126 + bg_linear_rgb[1] * 0.7152 + bg_linear_rgb[2] * 0.0722

	contrast_ratio_white_fg = white_fg_relative_luminance / (bg_relative_luminance + 0.05)
	contrast_ratio_black_fg = (bg_relative_luminance + 0.05) / black_fg_relative_luminance
	if contrast_ratio_white_fg > contrast_ratio_black_fg:
		# Return white
		return 1
	# Return black
	return 0

# Normalize bg_rgb to be within -1 and 1
def normalize_training_inputs(bg_rgb):
	return [(i - 0) / (255 - 0) for i in bg_rgb]

def generate_data():
	bg_rgb = get_rand_bg_color()
	normalized_bg_rgb = normalize_training_inputs(bg_rgb)
	label = get_optimum_fg_color(bg_rgb)
	return (normalized_bg_rgb, label)

def generate_dataset(total_batches, batch_size):
	dataset = []
	for i in range(total_batches):
		batch = []
		for j in range(batch_size):
			batch.append(generate_data())
		dataset.append(batch)
	return dataset

def test_dataset():
	batch = [([1, 1, 1], 0)]
	return [batch]

def generate_flat_dataset(num_datapoints):
	return [generate_data() for i in range(num_datapoints)]

def dump_dataset(dataset, filename):
	pickle.dump(dataset, open(filename, 'wb'))
	print("Dataset dumped.")

dump_dataset(generate_dataset(10, 32), dataset_filename)
