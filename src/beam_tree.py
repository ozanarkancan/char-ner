class Node(object):
    def __init__(self, memory, prob=1, preds = []):
        self.memory = memory
        self.prob = prob
        self.leaf = False
        self.preds = preds

class BeamTree(object):
    def __init__(self, beamsize):
        self.beamsize = beamsize
        self.beams = []
        self.cands = []
        self.all_leaf = False

    def prune(self):
        self.cands.sort(reverse=True, key=lambda x: x.prob)
        self.all_leaf = True
        self.beams = []

	limit = min(self.beamsize, len(self.cands))        

        for i in xrange(limit):
            self.beams.append(self.cands[i])
            self.all_leaf = self.all_leaf and self.cands[i].leaf

        self.cands = []

    def print_tree(self):
        print "Beams: "
        for n in self.beams:
            print "Node: "
            print n.prob
            print n.preds
