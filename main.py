from data_loader import MNISTLoader

test = MNISTLoader()
data = test.load_data()

print("data[0][0]", data[0][0])