from nltk import FreqDist, WittenBellProbDist

emissions = [('N', 'apple'), ('N', 'apple'), ('N', 'banana'), ('Adj', 'green'), ('V', 'sing')]
smoothed = {}
tags = set([t for (t,_) in emissions])
for tag in tags:
    words = [w for (t,w) in emissions if t == tag]
    smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)
print('smoothed probability of N -> apple is', smoothed['N'].prob('apple'))
print('smoothed probability of N -> banana is', smoothed['N'].prob('banana'))
print('smoothed probability of N -> peach is', smoothed['N'].prob('peach'))
print('smoothed probability of V -> sing is', smoothed['V'].prob('sing'))
print('smoothed probability of V -> walk is', smoothed['V'].prob('walk'))
