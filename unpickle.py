file = '"C:\Users\youk3\Documents\A-codeprojects\classifier\data\cifar-10-python.tar.gz"'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict