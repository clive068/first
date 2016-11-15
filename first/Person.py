class Person(object):
    def __init__(self, name):
        if len(name) > 0:
            self.name = name
        else:
            self.name = 'quotes'
            self.height = 120

    def height(self, param):
        self.height = param

    def getheight (self):
        return self.height

    def getAll (self):
        return (self.getName() + "'s height is " + str(self.getheight()))

    def getName(self):
        return self.name




if __name__ == '__main__':
    clive = Person('clive')
    clive.height(23)
    print (clive.getheight(), clive.getName())
    print clive.getAll()
    print (clive.__getattribute__('height'), clive.__getattribute__('name')) #works as well

    nicola = Person("")
    print nicola.getAll()




