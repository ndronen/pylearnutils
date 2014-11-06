class SentenceFile:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        return SentenceFileIterator(self.path)

class SentenceFileIterator:
    def __init__(self, path):
        self.f = open(path, "r")

    def __iter__(self):
        return self

    def next(self):
        line = self.f.readline()
        if line == '':
            raise StopIteration()
        return line.rstrip()
