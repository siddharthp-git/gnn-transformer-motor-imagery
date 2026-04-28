import pickle

file_path = '/Users/siddharth/Documents/Brainconnectivity/binary_label_0_vs_1/subject_0/explanation_trial_0.pkl'

# Open the file in binary mode ('rb' for read binary)
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Now 'data' contains the deserialized Python object
print(data)