from data_loader import MNISTLoader
# from neural_network import Network

test = MNISTLoader()
data = test.load_data()

print("data[0][0]", data[0][23])

# network_test = Network([2, 5, 1])

print("Shape of label:", data[0][23][1].shape)
print("Label values:", data[0][23][1].T)


# network_test.update_mini_batch(0, 0)