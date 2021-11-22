class random:

    def __init__(self, yo):
        self.yo = yo
        self.children = {"a": 0} 

    def print(self):
        # print(self[yo])
        for a in self.children:
            print(self[a])


if __name__ == "__main__":
    obj = random(2)
    obj.print()
    
